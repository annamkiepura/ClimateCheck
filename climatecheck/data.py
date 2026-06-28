"""Dataset loading and abstract-cache helpers."""

import os
import pickle

from datasets import load_dataset

from .config import (
    CLAIMS_DATASET_ID,
    ABSTRACTS_DATASET_ID,
    ABSTRACT_CACHE_DIR,
    ABSTRACT_CACHE_FILE,
)


def load_claims_train():
    return load_dataset(CLAIMS_DATASET_ID)["train"]


def load_claims_test():
    return load_dataset(CLAIMS_DATASET_ID)["test"]


def load_abstracts():
    return load_dataset(ABSTRACTS_DATASET_ID)["train"]


# ── Abstract cache ──────────────────────────────────────────────────────────

def build_abstract_cache(abstracts_dataset):
    """Extract texts, ids, and id-map from the abstracts dataset."""
    abstract_texts = [a["abstract"] for a in abstracts_dataset]
    abstract_ids = [a["abstract_id"] for a in abstracts_dataset]
    abstract_id_map = {i: aid for i, aid in enumerate(abstract_ids)}
    return abstract_texts, abstract_ids, abstract_id_map


def save_abstract_cache(abstract_texts, abstract_ids, abstract_id_map):
    os.makedirs(ABSTRACT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(ABSTRACT_CACHE_DIR, ABSTRACT_CACHE_FILE)
    cache = {
        "abstract_texts": abstract_texts,
        "abstract_ids": abstract_ids,
        "abstract_id_map": abstract_id_map,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved abstract cache to {cache_path}")


def load_abstract_cache():
    cache_path = os.path.join(ABSTRACT_CACHE_DIR, ABSTRACT_CACHE_FILE)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    return cache["abstract_texts"], cache["abstract_ids"], cache["abstract_id_map"]
