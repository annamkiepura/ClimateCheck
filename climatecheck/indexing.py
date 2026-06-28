"""Build BM25 and SPLADE retrieval indices."""

import pickle

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .config import BM25_CORPUS_PATH, SPLADE_INDEX_PATH, SPLADE_MODEL, DEVICE_STR


# ── BM25 index ──────────────────────────────────────────────────────────────

def build_bm25_index(abstract_texts):
    """Tokenize abstracts and save for BM25."""
    print("Building BM25 index...")
    abstract_tokenized = [word_tokenize(doc.lower()) for doc in tqdm(abstract_texts)]

    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(abstract_tokenized, f)

    print(f"Saved BM25 tokenized corpus to {BM25_CORPUS_PATH}")
    return abstract_tokenized


# ── SPLADE index ────────────────────────────────────────────────────────────

def build_splade_index(abstract_texts, batch_size=32, threshold=0.01):
    """Encode abstracts with SPLADE-v3 and save a sparse document-term matrix."""
    device = DEVICE_STR
    tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL)
    model.to(device).eval()

    vocab_size = tokenizer.vocab_size
    num_docs = len(abstract_texts)
    all_data, all_rows, all_cols = [], [], []

    for batch_start in tqdm(range(0, num_docs, batch_size), desc="Encoding SPLADE"):
        batch_texts = abstract_texts[batch_start : batch_start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc).logits
            weights = torch.relu(outputs)
            sparse_batch = weights.max(dim=1).values

        sparse_batch = sparse_batch.cpu().numpy()

        for i, vec in enumerate(sparse_batch):
            nz = np.where(vec > threshold)[0]
            all_rows.extend([batch_start + i] * len(nz))
            all_cols.extend(nz.tolist())
            all_data.extend(vec[nz].tolist())

    doc_term_matrix = csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(num_docs, vocab_size),
        dtype=np.float32,
    )

    save_npz(SPLADE_INDEX_PATH, doc_term_matrix)
    print(f"Saved SPLADE-v3 sparse index to {SPLADE_INDEX_PATH}")
    return doc_term_matrix
