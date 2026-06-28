#!/usr/bin/env python3
"""Fine-tune the BGE-M3 dense retriever and compute corpus embeddings.

Replaces: Fine-tune-dense.ipynb
"""

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from climatecheck.data import load_claims_train, load_abstract_cache
from climatecheck.dense_retriever import train_dense_retriever, compute_dense_embeddings


def main():
    print("=== Loading datasets ===")
    claims_dataset = load_claims_train()
    abstract_texts, _, _ = load_abstract_cache()

    print("\n=== Fine-tuning dense retriever ===")
    train_dense_retriever(claims_dataset)

    print("\n=== Computing dense embeddings ===")
    compute_dense_embeddings(abstract_texts)

    print("\nDone.")


if __name__ == "__main__":
    main()
