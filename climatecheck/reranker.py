"""Two-stage cross-encoder reranker fine-tuning and prediction."""

import random
import time
from collections import defaultdict

import pandas as pd
import torch
from datasets import Dataset, Value
from sentence_transformers import CrossEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from .config import CROSS_ENCODER_MODEL, RERANKER_STAGE1_DIR, RERANKER_STAGE2_DIR, DEVICE_STR

# ── Hyper-parameters ────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 200
K_HARD_NEG = 4
K_EASY_NEG = 4
INIT_EPOCHS = 3
RETRAIN_EPOCHS = 2
BATCH_SIZE = 16
TEST_SIZE = 0.1


def _build_lookups(claims_dataset, abstracts_dataset):
    """Build corpus dict, positives per claim, and non-relevant pool."""
    corpus = {rec["abstract_id"]: rec["abstract"] for rec in abstracts_dataset}
    positives = defaultdict(list)
    nonrel_pool = set()
    for rec in claims_dataset:
        cid = rec["claim_id"]
        ann = rec["annotation"].lower()
        if ann in ("supports", "refutes"):
            positives[cid].append(rec["abstract_id"])
        elif ann == "not enough information":
            nonrel_pool.add(rec["abstract_id"])
    return corpus, positives, nonrel_pool


def build_training_examples(claims_dataset, corpus, positives, nonrel_pool,
                            retrieve_fn, reranker=None):
    """
    Mine hard + easy negatives for cross-encoder training.

    Parameters
    ----------
    retrieve_fn : callable
        A function(query) -> list[(abstract_id, score)] that performs hybrid retrieval.
    reranker : CrossEncoder or None
        If provided, uses it to score candidates for harder negative mining.
    """
    examples = []
    for rec in tqdm(claims_dataset, desc="Mining examples"):
        cid, claim_text = rec["claim_id"], rec["claim"]
        if cid not in positives or not positives[cid]:
            continue

        # Retrieve fused top-K candidates
        top_hits = retrieve_fn(claim_text)
        candidates = [did for did, _ in top_hits]

        # Positives
        for pid in positives[cid]:
            examples.append({"claim": claim_text, "abstract": corpus[pid], "labels": 1.0})

        # Hard negatives
        if reranker:
            pairs = [(claim_text, corpus[did]) for did in candidates]
            scores = reranker.predict(pairs, batch_size=BATCH_SIZE)
            ranked = [did for _, did in sorted(zip(scores, candidates), reverse=True)]
            hard_ids = [did for did in ranked if did not in positives[cid] and did in nonrel_pool][:K_HARD_NEG]
        else:
            hard_ids = [did for did in candidates if did not in positives[cid] and did in nonrel_pool][:K_HARD_NEG]

        # Easy negatives
        easy_ids = random.sample(
            list(nonrel_pool - set(positives[cid])),
            k=min(K_EASY_NEG, len(nonrel_pool)),
        )

        for nid in hard_ids + easy_ids:
            examples.append({"claim": claim_text, "abstract": corpus[nid], "labels": 0.0})

    return Dataset.from_list(examples)


def preprocess_and_split(dataset):
    """Tokenize and split into train/test."""
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)

    def tokenize_fn(ex):
        return tokenizer(ex["claim"], ex["abstract"], truncation=True, padding=True)

    ds = dataset.map(tokenize_fn, batched=True)
    ds = ds.cast_column("labels", Value("float32"))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    split = ds.train_test_split(test_size=TEST_SIZE, seed=42)
    return split["train"], split["test"]


def train_reranker(train_ds, eval_ds, save_path, epochs):
    """Train a cross-encoder reranker and return it as a CrossEncoder."""
    model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL).to(DEVICE_STR)
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)

    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        logging_steps=100,
        report_to="none",
        metric_for_best_model="recall",
        greater_is_better=True,
    )

    def compute_metrics(p):
        scores = p.predictions[:, 0]
        preds = (scores > 0.5).astype(int)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="macro"),
            "precision": precision_score(p.label_ids, preds, average="macro"),
            "recall": recall_score(p.label_ids, preds, average="macro"),
            "log_loss": log_loss(p.label_ids, scores),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(save_path)
    return CrossEncoder(save_path, device=DEVICE_STR)


def run_two_stage_training(claims_dataset, abstracts_dataset, retrieve_fn):
    """
    Full two-stage reranker fine-tuning pipeline.

    Parameters
    ----------
    retrieve_fn : callable
        A function(query) -> list[(abstract_id, score)].
    """
    corpus, positives, nonrel_pool = _build_lookups(claims_dataset, abstracts_dataset)

    # Stage 1
    print("=== Stage 1: initial dataset ===")
    init_ds = build_training_examples(
        claims_dataset, corpus, positives, nonrel_pool, retrieve_fn, reranker=None,
    )
    init_ds.save_to_disk("./init_ds_FIXED")
    train_ds, eval_ds = preprocess_and_split(init_ds)
    reranker = train_reranker(train_ds, eval_ds, RERANKER_STAGE1_DIR, INIT_EPOCHS)

    # Stage 2
    print("=== Stage 2: refined dataset ===")
    refined_ds = build_training_examples(
        claims_dataset, corpus, positives, nonrel_pool, retrieve_fn, reranker=reranker,
    )
    refined_ds.save_to_disk("./refined_ds_FIXED")
    train_ds2, eval_ds2 = preprocess_and_split(refined_ds)
    reranker2 = train_reranker(train_ds2, eval_ds2, RERANKER_STAGE2_DIR, RETRAIN_EPOCHS)

    print("Two-stage fine-tuning complete.")
    return reranker2


def predict_with_reranker(verification_dataset, abstract_ids, abstracts_dataset,
                          retrieve_fn, reranker_path=None):
    """Generate top-10 ranked predictions for each verification claim."""
    reranker_path = reranker_path or RERANKER_STAGE2_DIR
    reranker = CrossEncoder(reranker_path, device=DEVICE_STR)
    abstract_corpus = {a["abstract_id"]: a["abstract"] for a in abstracts_dataset}

    results = []
    for claim in tqdm(verification_dataset, desc="Classifying"):
        claim_id = claim["claim_id"]
        claim_text = claim["claim"]

        top_rrf = retrieve_fn(claim_text)

        pairs = [[claim_text, abstract_corpus[abs_id]] for abs_id, _ in top_rrf]
        scores = reranker.predict(pairs, show_progress_bar=True)

        ranked = sorted(zip(top_rrf, scores), key=lambda x: x[1], reverse=True)
        top_10 = ranked[:10]

        for rank, ((abs_id, _), rerank_score) in enumerate(top_10):
            results.append({
                "claim_id": claim_id,
                "abstract_id": abs_id,
                "rank": rank + 1,
            })

    df = pd.DataFrame(results)
    df.to_csv("predictions_FIXED.csv", index=False)
    print("Saved predictions to predictions_FIXED.csv")
    return df
