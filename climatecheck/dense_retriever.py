"""Fine-tune BGE-M3 dense retriever with triplet loss and compute embeddings."""

import gc
import gzip
import json
import math
import pickle
import random
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    LoggingHandler,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import TripletDistanceMetric
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BGE_M3_MODEL,
    DENSE_MODEL_DIR,
    DENSE_EMBEDDINGS_PATH,
    TRIPLETS_DIR,
    SEED,
    DEVICE_STR,
)


# ── Hyper-parameters ────────────────────────────────────────────────────────
SPLIT_RATIO = 0.1
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 32
LR = 5e-5
MARGIN = 0.3
NEGATIVE_POOL = 20
EVAL_INTERVAL = 100
K_VALS = [1, 5, 10]


# ── Triplet helpers ─────────────────────────────────────────────────────────

def sample_triplets(dataset, nei_pool, k_neg=1):
    """Create InputExample triplets from dataset iterator."""
    triplets = []
    for rec in dataset:
        label = rec["annotation"].lower()
        if label in ("supports", "refutes"):
            pos = rec["abstract"]
            negatives = random.sample(nei_pool, k=k_neg)
            for neg in negatives:
                triplets.append(InputExample(texts=[rec["claim"], pos, neg], label=0.0))
    return triplets


def dump_triplets(triplets, filepath):
    """Write triplets to newline-delimited JSON."""
    filepath = Path(filepath)
    with filepath.open("w", encoding="utf8") as f:
        for ex in triplets:
            anchor, pos, neg = ex.texts
            json.dump({"anchor": anchor, "positive": pos, "negative": neg}, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(triplets):,} triplets -> {filepath}")


# ── Training ────────────────────────────────────────────────────────────────

def train_dense_retriever(claims_dataset):
    """Fine-tune BGE-M3 on triplet loss and save the model."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    gc.collect()
    torch.cuda.empty_cache()

    triplets_dir = Path(TRIPLETS_DIR)
    triplets_dir.mkdir(exist_ok=True)

    # 1. Split data
    raw_ds = claims_dataset
    labels = [lbl.lower() for lbl in raw_ds["annotation"]]

    train_idx, eval_idx = train_test_split(
        list(range(len(raw_ds))),
        test_size=SPLIT_RATIO,
        stratify=labels,
        random_state=SEED,
    )
    train_ds = raw_ds.select(train_idx)
    eval_ds = raw_ds.select(eval_idx)
    print(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # 2. Build triplets
    nei_pool_train = [r["abstract"] for r in train_ds if r["annotation"].lower().startswith("not")]
    nei_pool_eval = [r["abstract"] for r in eval_ds if r["annotation"].lower().startswith("not")]

    print("Creating train triplets...")
    train_triplets = sample_triplets(train_ds, nei_pool_train, k_neg=1)
    dump_triplets(train_triplets, triplets_dir / "train_triplets.jsonl")

    print("Creating eval triplets...")
    eval_triplets = sample_triplets(eval_ds, nei_pool_eval, k_neg=1)
    dump_triplets(eval_triplets, triplets_dir / "eval_triplets.jsonl")

    # 3. Model, dataloader, loss
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    model = SentenceTransformer(BGE_M3_MODEL, device=DEVICE_STR)

    train_loader = DataLoader(train_triplets, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    triplet_loss = losses.TripletLoss(
        model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=MARGIN,
    )

    # 4. IR evaluator
    qrels = defaultdict(set)
    queries = {}
    corpus = {}

    for t in eval_triplets:
        anchor, pos, _ = t.texts
        qid = hash(anchor)
        docid = hash(pos)
        queries[qid] = anchor
        corpus[docid] = pos
        qrels[qid].add(docid)

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, qrels,
        mrr_at_k=K_VALS,
        precision_recall_at_k=K_VALS,
        show_progress_bar=True,
        name="eval",
    )

    # 5. Train
    warmup_steps = math.ceil(len(train_loader) * EPOCHS * 0.1)

    model.fit(
        train_objectives=[(train_loader, triplet_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        use_amp=True,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=EVAL_INTERVAL,
        output_path=DENSE_MODEL_DIR,
        save_best_model=True,
    )

    print(f"Fine-tuned model saved to {Path(DENSE_MODEL_DIR).resolve()}")
    return model


# ── Embedding computation ──────────────────────────────────────────────────

def compute_dense_embeddings(abstract_texts, model_dir=None, batch_size=2):
    """Encode the full abstract corpus with the fine-tuned BGE-M3 model."""
    model_dir = model_dir or DENSE_MODEL_DIR
    gc.collect()
    torch.cuda.empty_cache()

    print("Encoding corpus with fine-tuned BGE-M3...")
    bge_model = SentenceTransformer(model_dir, device=DEVICE_STR)

    emb_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(abstract_texts), batch_size)):
            batch_texts = abstract_texts[i : i + batch_size]
            emb = bge_model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            emb_chunks.append(emb)
            torch.cuda.empty_cache()

    bge_vecs = np.vstack(emb_chunks)
    print(f"Done: {bge_vecs.shape}")

    dense_embeddings = {"bge": bge_vecs}
    with gzip.open(DENSE_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(dense_embeddings, f)

    print(f"Saved merged embeddings -> {DENSE_EMBEDDINGS_PATH}")
    return dense_embeddings
