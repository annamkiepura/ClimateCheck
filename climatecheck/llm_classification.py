"""GPT-4 based claim classification: 0-shot, 2-stage, and hybrid approaches."""

import re

import pandas as pd
from openai import OpenAI

from .config import OPENAI_MODEL
from .fewshot_examples import LLM_CLASSIFICATION_FEWSHOT_EXAMPLES, LLM_FEWSHOT_VARIANT_EXAMPLES


def _safe(x):
    """Convert NaNs etc. to empty strings so we always pass text."""
    return "" if pd.isna(x) else str(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Strategy A: 0-shot 1-stage
# ═════════════════════════════════════════════════════════════════════════════

_SYSTEM_0SHOT_1STAGE = (
    "You are an expert scientific fact-checker.\n"
    "Given a claim and a paper abstract, reply with exactly one of:\n"
    "supports | refutes | not enough information"
)


def classify_0shot_1stage(df, api_key):
    """Apply 0-shot 1-stage classification. Returns df with 'label' column."""
    client = OpenAI(api_key=api_key)

    def _classify(row):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_0SHOT_1STAGE},
                {
                    "role": "user",
                    "content": f"Claim:\n{_safe(row.claim)}\n\nAbstract:\n{_safe(row.abstract)}",
                },
            ],
        )
        label = response.choices[0].message.content.strip().lower()
        legit = {"supports", "refutes", "not enough information"}
        if label not in legit:
            raise ValueError(f"unexpected label \u00ab{label}\u00bb from model")
        return label

    df["label"] = df.apply(_classify, axis=1)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Strategy B: 0-shot 2-stages
# ═════════════════════════════════════════════════════════════════════════════

_SYSTEM_STAGE1 = (
    "You are an expert scientific fact-checker.\n"
    "Task\n"
    "-----\n"
    "Given one claim and one scientific-paper abstract, decide whether the abstract "
    "contains evidence that directly supports OR directly refutes the claim.\n\n"
    "Label definitions\n"
    "\u2022 EVIDENCE \u2013 The abstract presents data, observations, arguments, or findings that clearly "
    "support *or* contradict the claim. Mere topical overlap is insufficient; there "
    "must be an evidentiary link.\n"
    "\u2022 UNKNOWN \u2013 \u201cNot enough information.\u201d The abstract is off-topic, only tangentially "
    "related, or lacks evidence about the claim\u2019s truth value.\n\n"
    "Output rules\n"
    "-------------\n"
    "1. Think silently and completely before answering.\n"
    "2. Then output exactly one of the two uppercase tokens, with no extra words, "
    "punctuation, or whitespace: EVIDENCE or UNKNOWN\n"
    "3. If the inputs are missing or malformed, output UNKNOWN.\n\n"
    "You must never reveal your reasoning\u2014only the single label."
)

_SYSTEM_STAGE2 = (
    "You are an expert scientific fact-checker.\n"
    "Task\n"
    "-----\n"
    "Given one claim and one scientific-paper abstract, decide whether the abstract "
    "contains evidence that directly supports OR directly refutes the claim.\n\n"
    "Label definitions\n"
    "\u2022 supports \u2013 The abstract presents data, observations, arguments, or findings that clearly support the claim.\n"
    "\u2022 refutes \u2013 The abstract presents data, observations, arguments, or findings that clearly contradict the claim.\n"
    "Mere topical overlap is insufficient; there must be an evidentiary link.\n\n"
    "Output rules\n"
    "-------------\n"
    "1. Think silently and completely before answering.\n"
    "2. Then output exactly one of the two lowercase tokens, with no extra words, "
    "punctuation, or whitespace: supports or refutes\n"
    "3. If the inputs are missing or malformed, output UNKNOWN.\n\n"
    "You must never reveal your reasoning\u2014only the single label."
)


def classify_0shot_2stage(df, api_key):
    """Apply 0-shot 2-stage classification. Returns df with 'label' column."""
    client = OpenAI(api_key=api_key)

    # Stage 1: EVIDENCE vs UNKNOWN
    def _stage1(row):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            top_p=1,
            max_tokens=3,
            messages=[
                {"role": "system", "content": _SYSTEM_STAGE1},
                {
                    "role": "user",
                    "content": f"Claim:\n{_safe(row.claim)}\n\nAbstract:\n{_safe(row.abstract)}",
                },
            ],
        )
        label = response.choices[0].message.content.strip().upper()
        legit = {"EVIDENCE", "UNKNOWN"}
        if label not in legit:
            raise ValueError(f"Unexpected label \u00ab{label}\u00bb from model")
        return label

    df["label"] = df.apply(_stage1, axis=1)
    df.to_csv("predictions_stage_1.csv", index=False)

    # Stage 2: supports vs refutes (only for EVIDENCE rows)
    def _stage2(row):
        if row.label == "UNKNOWN":
            return row.label
        if row.label == "EVIDENCE":
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                top_p=1,
                max_tokens=3,
                messages=[
                    {"role": "system", "content": _SYSTEM_STAGE2},
                    {
                        "role": "user",
                        "content": f"Claim:\n{_safe(row.claim)}\n\nAbstract:\n{_safe(row.abstract)}",
                    },
                ],
            )
            raw = response.choices[0].message.content
            label = raw.strip().lower()
            legit = {"supports", "refutes"}
            if label not in legit:
                raise ValueError(f"Unexpected label \u00ab{label}\u00bb from model")
            return label
        return row.label

    df["label"] = df.apply(_stage2, axis=1)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Strategy C/D: Hybrid (0-shot or few-shot)
# ═════════════════════════════════════════════════════════════════════════════

_SYSTEM_HYBRID = (
    "You are an expert scientific fact-checker.\n\n"
    "Task\n"
    "-----\n"
    "For a given claim and one paper abstract, reason internally in two steps:\n"
    "  1. Decide if the abstract contains evidence that directly supports OR directly refutes the claim.\n"
    "  2. If evidence exists, decide whether it SUPPORTS or REFUTES.\n\n"
    "Output rules\n"
    "-------------\n"
    "\u2022 Think silently; do NOT reveal your reasoning.\n"
    "\u2022 Then output **exactly one** of these uppercase tokens with nothing else:\n"
    "    SUPPORTS   (evidence backs the claim)\n"
    "    REFUTES    (evidence contradicts the claim)\n"
    "    NEI        (Not Enough Information \u2013 no evidence)\n"
    "\u2022 If the input is malformed, your output is irrelevant because the client will never ask you "
    "(inputs are pre-validated)."
)

LEGIT_HYBRID = {"SUPPORTS", "REFUTES", "NEI"}


def _normalise(text):
    """Strip whitespace/punct, uppercase -> pure label."""
    return re.sub(r"[^A-Z]", "", text.upper())


def _validate_inputs(claim, abstract, row_idx=None):
    if not (isinstance(claim, str) and claim.strip()):
        raise ValueError(f"Row {row_idx}: empty or non-string claim")
    if not (isinstance(abstract, str) and abstract.strip()):
        raise ValueError(f"Row {row_idx}: empty or non-string abstract")


def _build_fewshot_messages(fewshot_examples):
    """Build reusable chat messages from few-shot examples."""
    messages = []
    for ex in fewshot_examples:
        claim = ex["claim"]
        for item in ex["items"]:
            messages.append(
                {"role": "user", "content": f"Claim:\n{claim}\n\nAbstract:\n{item['text']}"}
            )
            messages.append({"role": "assistant", "content": item["label"].upper()})
    return messages


def classify_hybrid(df, api_key, use_fewshot=False):
    """
    Apply hybrid classification (0-shot or few-shot).

    Parameters
    ----------
    use_fewshot : bool
        If True, use few-shot examples; otherwise 0-shot.
    """
    client = OpenAI(api_key=api_key)

    if use_fewshot:
        fewshot_messages = _build_fewshot_messages(LLM_FEWSHOT_VARIANT_EXAMPLES)
    else:
        fewshot_messages = _build_fewshot_messages(LLM_CLASSIFICATION_FEWSHOT_EXAMPLES)

    def _build_messages(claim, abstract):
        msgs = [{"role": "system", "content": _SYSTEM_HYBRID}]
        if use_fewshot:
            msgs.extend(fewshot_messages)
        msgs.append({"role": "user", "content": f"Claim:\n{claim}\n\nAbstract:\n{abstract}"})
        return msgs

    def _classify(row):
        _validate_inputs(row.claim, row.abstract, row.name)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            top_p=1,
            max_tokens=3,
            messages=_build_messages(row.claim, row.abstract),
        )
        raw = response.choices[0].message.content
        label = _normalise(raw)
        if label not in LEGIT_HYBRID:
            raise ValueError(f"Unexpected label \u00ab{raw}\u00bb (\u2192 {label})")
        return label

    df["label"] = df.apply(_classify, axis=1)

    # Map NEI -> standard label
    df["label"] = df["label"].replace("NEI", "not enough information")
    return df
