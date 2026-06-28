"""RankGPT-based LLM ranking combined with cross-encoder reranking."""

import importlib
from typing import Dict, List

import pandas as pd
import rank_gpt
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from .config import OPENAI_MODEL, RERANKER_STAGE2_DIR, DEVICE_STR
from .fewshot_examples import RANKGPT_FEWSHOT_EXAMPLES

# ── Configuration ───────────────────────────────────────────────────────────
ALPHA = 0.4  # 0 = only LLM, 1 = only cross-encoder


# ── Few-shot helpers (permutation) ──────────────────────────────────────────

def build_perm_fewshots(fewshots: List[dict]) -> List[dict]:
    """
    Turn few-shot examples into a message list for ChatGPT:
    - user message  -> claim + numbered passages
    - assistant msg -> best permutation (evidence first)
    """
    messages = []
    for ex in fewshots:
        claim = ex["claim"]
        evid = [p for p in ex["items"] if p["label"] in ("supports", "refutes")]
        nei = [p for p in ex["items"] if p["label"] == "nei"]
        items = evid + nei
        text_block = "\n".join(f"{i+1}) {p['text']}" for i, p in enumerate(items))
        messages.append({
            "role": "user",
            "content": (
                f"Claim: {claim}\nPassages:\n{text_block}\n"
                "Respond with the best permutation of the passage numbers "
                "from most evidentiary to least."
            ),
        })
        permutation = " ".join(str(i + 1) for i in range(len(items)))
        messages.append({"role": "assistant", "content": permutation})
    return messages


CUSTOM_PERM_SYS = (
    "You are given a CLAIM and N PASSAGES. "
    "A passage is *evidentiary* with respect to a claim if it contains information that could "
    "either SUPPORT or REFUTE the claim. Whether it supports or refutes "
    "does not matter. Return exactly one line with the passage numbers, most "
    "evidentiary first, least evidentiary last. Output numbers only."
)


def patch_rankgpt(fewshot_messages: List[dict]):
    """Monkey-patch RankGPT to prepend few-shot exemplars to permutation prompts."""
    rank_gpt_mod = importlib.reload(rank_gpt)

    if not hasattr(rank_gpt_mod, "_orig_create_perm"):
        rank_gpt_mod._orig_create_perm = rank_gpt_mod.create_permutation_instruction

    def create_perm_with_fewshot(item=None, **kw):
        msgs = rank_gpt_mod._orig_create_perm(item=item, **kw)
        msgs = [m for m in msgs if m.get("role") != "system"]
        return (
            fewshot_messages
            + [{"role": "system", "content": CUSTOM_PERM_SYS}]
            + msgs
        )

    rank_gpt_mod.create_permutation_instruction = create_perm_with_fewshot
    return rank_gpt_mod


# ── Score fusion ────────────────────────────────────────────────────────────

def fuse_scores(
    rank_positions: Dict[int, int],
    ce_scores: Dict[int, float],
    alpha: float = ALPHA,
) -> Dict[int, float]:
    """
    Combine LLM permutation (rank 1 = best) with cross-encoder scores.

    fused_score = alpha * norm_ce + (1 - alpha) * norm_llm
    """
    min_ce, max_ce = min(ce_scores.values()), max(ce_scores.values())
    norm_ce = {
        k: 0.0 if max_ce == min_ce else (v - min_ce) / (max_ce - min_ce)
        for k, v in ce_scores.items()
    }

    max_rank = max(rank_positions.values())
    norm_llm = {
        k: 1 - (r - 1) / (max_rank - 1) if max_rank > 1 else 1.0
        for k, r in rank_positions.items()
    }

    return {k: alpha * norm_ce[k] + (1 - alpha) * norm_llm[k] for k in rank_positions}


# ── Main ranking loop ──────────────────────────────────────────────────────

def run_evidentiary_ranking_loop(
    verification_dataset,
    abstract_corpus,
    retrieve_fn,
    api_key,
    *,
    reranker_path=None,
    top_k_retrieval=600,
    top_k_reranker=20,
    final_k=10,
    alpha=ALPHA,
) -> pd.DataFrame:
    """
    For each verification claim:
    1. Hybrid retrieval
    2. Cross-encoder rerank
    3. RankGPT permutation
    4. Score fusion
    5. Select top-k

    Returns DataFrame with columns: claim_id, abstract_id, rank
    """
    reranker_path = reranker_path or RERANKER_STAGE2_DIR
    reranker = CrossEncoder(reranker_path, device=DEVICE_STR)

    # Patch RankGPT with few-shot examples
    perm_fewshots = build_perm_fewshots(RANKGPT_FEWSHOT_EXAMPLES)
    rank_gpt_mod = patch_rankgpt(perm_fewshots)

    rows = []
    for rec in tqdm(verification_dataset, desc="Processing claims"):
        cid, claim = rec["claim_id"], rec["claim"]

        # 1. Hybrid retrieval
        top_retrieval = retrieve_fn(claim)

        # 2. Cross-encoder rerank
        cand_ids = [aid for aid, _ in top_retrieval]
        pairs = [[claim, abstract_corpus[aid]] for aid in cand_ids]
        ce_scores_raw = reranker.predict(pairs, show_progress_bar=False)

        top_rerank = sorted(zip(cand_ids, ce_scores_raw), key=lambda x: -x[1])[:top_k_reranker]
        ce_scores_map = {aid: s for aid, s in top_rerank}

        # 3. RankGPT permutation
        item_for_sw = {
            "query": claim,
            "hits": [
                {"docid": aid, "content": abstract_corpus[aid], "score": float(score)}
                for aid, score in top_rerank
            ],
        }

        permuted = rank_gpt_mod.sliding_windows(
            item_for_sw,
            rank_start=0,
            rank_end=len(item_for_sw["hits"]),
            window_size=len(item_for_sw["hits"]),
            step=len(item_for_sw["hits"]),
            model_name=OPENAI_MODEL,
            api_key=api_key,
        )

        llm_rank_pos = {hit["docid"]: i + 1 for i, hit in enumerate(permuted["hits"])}

        # 4. Score fusion
        fused = fuse_scores(llm_rank_pos, ce_scores_map, alpha=alpha)

        # 5. Select top-k
        ordered_ids = sorted(fused, key=fused.get, reverse=True)[:final_k]

        for rank, aid in enumerate(ordered_ids, start=1):
            rows.append({"claim_id": cid, "abstract_id": aid, "rank": rank})

    df = pd.DataFrame(rows, columns=["claim_id", "abstract_id", "rank"])
    return df
