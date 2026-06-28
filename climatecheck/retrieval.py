"""Hybrid retrieval: BM25 + SPLADE + Dense, fused with Reciprocal Rank Fusion."""

import os
import gzip
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .config import (
    BM25_CORPUS_PATH,
    SPLADE_INDEX_PATH,
    SPLADE_MODEL,
    DENSE_EMBEDDINGS_PATH,
    DENSE_MODEL_DIR,
    DEVICE_STR,
)


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_bm25(abstract_texts=None):
    """Load (or build + cache) BM25 from tokenized abstracts."""
    if os.path.exists(BM25_CORPUS_PATH):
        with open(BM25_CORPUS_PATH, "rb") as f:
            tokenized_abstracts = pickle.load(f)
    else:
        from tqdm import tqdm
        tokenized_abstracts = [
            word_tokenize(doc.lower()) for doc in tqdm(abstract_texts, desc="Tokenizing")
        ]
        with open(BM25_CORPUS_PATH, "wb") as f:
            pickle.dump(tokenized_abstracts, f)
    return BM25Okapi(tokenized_abstracts)


def load_splade():
    """Load the SPLADE model, tokenizer, and pre-computed document matrix."""
    splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL)
    splade_model.to(DEVICE_STR).eval()
    doc_term_matrix = load_npz(SPLADE_INDEX_PATH)
    print(f"Loaded SPLADE-v3 sparse index from {SPLADE_INDEX_PATH}")
    return splade_model, splade_tokenizer, doc_term_matrix


def load_dense_embeddings(path=None):
    """Load gzip-compressed pickle of dense embeddings."""
    path = Path(path or DENSE_EMBEDDINGS_PATH)
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rb") as f:
        return pickle.load(f)


def load_dense_retrievers(model_dir=None):
    """Return dict of dense SentenceTransformer retrievers."""
    model_dir = model_dir or DENSE_MODEL_DIR
    return {
        "bge": SentenceTransformer(model_dir).to(DEVICE_STR),
    }


# ── SPLADE-only retrieval ──────────────────────────────────────────────────

def retrieve_splade_only(query, splade_model, splade_tokenizer, doc_term_matrix,
                         abstract_ids, top_k=200):
    enc = splade_tokenizer(
        [query], return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(next(splade_model.parameters()).device)

    with torch.no_grad():
        logits = splade_model(**enc).logits
        weights = torch.relu(logits)
        q_vec = weights.max(dim=1).values.squeeze(0).cpu().numpy()

    scores = doc_term_matrix.dot(q_vec)
    top_idxs = np.argpartition(-scores, top_k)[:top_k]
    top_sorted = top_idxs[np.argsort(-scores[top_idxs])]
    return [abstract_ids[i] for i in top_sorted]


# ── Hybrid retrieval with RRF fusion ───────────────────────────────────────

def retrieve_with_all(
    query,
    bm25,
    corpus_ids,
    retrievers,
    dense_embeddings,
    splade_model,
    splade_tokenizer,
    splade_doc_term_matrix,
    abstract_ids,
    top_k=200,
    k_rrf=60,
):
    """
    Retrieve using BM25 + SPLADE + Dense and fuse with Reciprocal Rank Fusion.

    Returns list of (abstract_id, fused_score) sorted by score descending.
    """
    rank_lists = []

    # BM25
    tokens = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokens)
    bm25_top = np.argsort(bm25_scores)[-top_k:][::-1]
    rank_lists.append([corpus_ids[i] for i in bm25_top])

    # SPLADE
    splade_top_ids = retrieve_splade_only(
        query, splade_model, splade_tokenizer, splade_doc_term_matrix,
        abstract_ids, top_k=top_k,
    )
    rank_lists.append(splade_top_ids)

    # Dense
    for name, model in retrievers.items():
        q_vec_dense = model.encode([query], normalize_embeddings=True)
        scores = dense_embeddings[name] @ q_vec_dense.T
        top_i = np.argsort(scores.ravel())[-top_k:][::-1]
        rank_lists.append([corpus_ids[i] for i in top_i])

    # RRF fusion
    fused = defaultdict(float)
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            fused[doc_id] += 1.0 / (k_rrf + rank + 1)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
