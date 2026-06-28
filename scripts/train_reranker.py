#!/usr/bin/env python3
"""Two-stage cross-encoder reranker training and prediction.

Replaces: fine-tune-CE.ipynb
"""

import gc

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import torch

from climatecheck.data import load_claims_train, load_claims_test, load_abstracts, load_abstract_cache
from climatecheck.retrieval import (
    load_bm25,
    load_splade,
    load_dense_embeddings,
    load_dense_retrievers,
    retrieve_with_all,
)
from climatecheck.reranker import run_two_stage_training, predict_with_reranker


def main():
    print("=== Loading datasets ===")
    claims_dataset = load_claims_train()
    abstracts_dataset = load_abstracts()
    verification_dataset = load_claims_test()
    abstract_texts, abstract_ids, abstract_id_map = load_abstract_cache()

    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Loading retrieval components ===")
    bm25 = load_bm25(abstract_texts)
    splade_model, splade_tokenizer, doc_term_matrix = load_splade()
    dense_embeddings = load_dense_embeddings()
    retrievers = load_dense_retrievers()

    corpus_ids = list({rec["abstract_id"] for rec in abstracts_dataset})

    def retrieve_fn(query, top_k=200):
        return retrieve_with_all(
            query,
            bm25=bm25,
            corpus_ids=corpus_ids,
            retrievers=retrievers,
            dense_embeddings=dense_embeddings,
            splade_model=splade_model,
            splade_tokenizer=splade_tokenizer,
            splade_doc_term_matrix=doc_term_matrix,
            abstract_ids=abstract_ids,
            top_k=top_k,
        )

    print("\n=== Training reranker (two-stage) ===")
    run_two_stage_training(claims_dataset, abstracts_dataset, retrieve_fn)

    print("\n=== Generating predictions ===")
    predict_with_reranker(
        verification_dataset, abstract_ids, abstracts_dataset, retrieve_fn,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
