#!/usr/bin/env python3
"""Build BM25 and SPLADE retrieval indices, and cache abstract metadata.

Replaces: Make_indices.ipynb
"""

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from climatecheck.data import (
    load_abstracts,
    build_abstract_cache,
    save_abstract_cache,
)
from climatecheck.indexing import build_bm25_index, build_splade_index


def main():
    print("=== Loading abstracts dataset ===")
    abstracts_dataset = load_abstracts()

    abstract_texts, abstract_ids, abstract_id_map = build_abstract_cache(abstracts_dataset)

    print("\n=== Building BM25 index ===")
    build_bm25_index(abstract_texts)

    print("\n=== Building SPLADE index ===")
    build_splade_index(abstract_texts)

    print("\n=== Saving abstract cache ===")
    save_abstract_cache(abstract_texts, abstract_ids, abstract_id_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
