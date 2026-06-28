#!/usr/bin/env python3
"""RankGPT-based LLM ranking combined with cross-encoder reranking.

Replaces: RankGPT.ipynb

Requires OPENAI_API_KEY environment variable to be set.
"""

import gc
import os

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import torch

from climatecheck.data import load_claims_test, load_abstracts, load_abstract_cache
from climatecheck.retrieval import (
    load_bm25,
    load_splade,
    load_dense_embeddings,
    load_dense_retrievers,
    retrieve_with_all,
)
from climatecheck.rankgpt_ranking import run_evidentiary_ranking_loop


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable")

    print("=== Loading datasets ===")
    abstracts_dataset = load_abstracts()
    verification_dataset = load_claims_test()
    abstract_texts, abstract_ids, _ = load_abstract_cache()

    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Loading retrieval components ===")
    bm25 = load_bm25(abstract_texts)
    splade_model, splade_tokenizer, doc_term_matrix = load_splade()
    dense_embeddings = load_dense_embeddings()
    retrievers = load_dense_retrievers()

    abstract_corpus = {a["abstract_id"]: a["abstract"] for a in abstracts_dataset}
    corpus_ids = list(abstract_corpus.keys())

    def retrieve_fn(query, top_k=600):
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

    print("\n=== Running RankGPT ranking ===")
    df = run_evidentiary_ranking_loop(
        verification_dataset,
        abstract_corpus,
        retrieve_fn,
        api_key,
    )

    df.to_csv("verification_predictions.csv", index=False)
    print(f"\nSaved {len(df)} rows to verification_predictions.csv")
    print("Done.")


if __name__ == "__main__":
    main()
