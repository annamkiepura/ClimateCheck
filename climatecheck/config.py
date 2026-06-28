"""Shared paths, constants, and device setup."""

import os
import torch

os.environ["WANDB_DISABLED"] = "true"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# ── Artifact paths ──────────────────────────────────────────────────────────
BM25_CORPUS_PATH = "bm25_abstract_tokenized.pkl"
SPLADE_INDEX_PATH = "splade_v3_docs.npz"
ABSTRACT_CACHE_DIR = "./abstract_variables/"
ABSTRACT_CACHE_FILE = "abstract_cache.pkl"
DENSE_EMBEDDINGS_PATH = "dense_embeddings_ft.pkl.gz"
DENSE_MODEL_DIR = "bge-m3-finetuned-triplet"
RERANKER_STAGE1_DIR = "./model_stage1_FIXED"
RERANKER_STAGE2_DIR = "./model_stage2_FIXED"
CLASSIFIER_DIR = "deberta-claim-checker-mnli"
TRIPLETS_DIR = "triplets"

# ── Model names ─────────────────────────────────────────────────────────────
BGE_M3_MODEL = "BAAI/bge-m3"
SPLADE_MODEL = "naver/splade-v3"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEBERTA_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli"
OPENAI_MODEL = "gpt-4.1"

# ── HuggingFace dataset IDs ────────────────────────────────────────────────
CLAIMS_DATASET_ID = "rabuahmad/climatecheck"
ABSTRACTS_DATASET_ID = "rabuahmad/climatecheck_publications_corpus"

# ── Common hyper-parameters ─────────────────────────────────────────────────
SEED = 42
