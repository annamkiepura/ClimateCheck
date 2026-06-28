# ClimateCheck

A climate claim verification pipeline that checks whether scientific abstracts **support**, **refute**, or provide **not enough information** about a given climate-related claim.

The system combines multiple retrieval, reranking, and classification strategies to match claims against a corpus of scientific publication abstracts.

## Pipeline overview

```
Claims + Abstract Corpus
         |
    1. Indexing            (BM25, SPLADE-v3)
         |
    2. Dense Retriever     (fine-tuned BGE-M3 with triplet loss)
         |
    3. Hybrid Retrieval    (BM25 + SPLADE + Dense, fused with RRF)
         |
    4. Reranking           (two-stage cross-encoder fine-tuning)
         |
    5. Classification      (GPT-4 / fine-tuned DeBERTa / RankGPT)
         |
    Predictions CSV
```

## Project structure

```
climatecheck/              # Python package with reusable modules
    __init__.py
    config.py              # Shared paths, constants, model names
    data.py                # Dataset loading and abstract cache helpers
    indexing.py            # BM25 and SPLADE index construction
    dense_retriever.py     # BGE-M3 fine-tuning and embedding computation
    retrieval.py           # Hybrid retrieval (BM25 + SPLADE + Dense + RRF)
    reranker.py            # Two-stage cross-encoder reranker
    rankgpt_ranking.py     # RankGPT LLM-based ranking with score fusion
    llm_classification.py  # GPT-4 classification (0-shot, 2-stage, hybrid)
    ft_classification.py   # DeBERTa fine-tuned classifier with focal loss
    fewshot_examples.py    # Shared few-shot examples for ranking/classification

scripts/                   # Entry-point scripts (one per pipeline stage)
    build_indices.py       # Step 1: Build BM25 + SPLADE indices
    train_dense_retriever.py  # Step 2: Fine-tune dense retriever + compute embeddings
    train_reranker.py      # Step 3: Train two-stage cross-encoder reranker
    run_rankgpt.py         # Step 4a: RankGPT-based ranking
    run_llm_classification.py  # Step 4b: GPT-4 classification
    run_ft_classification.py   # Step 4c: Fine-tuned DeBERTa classification

requirements.txt
README.md
```

## Setup

```bash
pip install -r requirements.txt
```

For SPLADE model access, log in to HuggingFace:

```bash
huggingface-cli login
```

For GPT-4 based classification and RankGPT, set the OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Data

All datasets are loaded from HuggingFace Hub:

| Dataset | HuggingFace ID | Split |
|---------|---------------|-------|
| Claims (train) | `rabuahmad/climatecheck` | `train` |
| Claims (test) | `rabuahmad/climatecheck` | `test` |
| Abstract corpus | `rabuahmad/climatecheck_publications_corpus` | `train` |

Each claim has an annotation: `supports`, `refutes`, or `not enough information`.

## Usage

Run the pipeline stages in order. Each script corresponds to one of the original notebooks.

### Step 1: Build retrieval indices

Builds BM25 tokenized corpus, SPLADE-v3 sparse document-term matrix, and caches abstract metadata.

```bash
python scripts/build_indices.py
```

**Outputs:** `bm25_abstract_tokenized.pkl`, `splade_v3_docs.npz`, `abstract_variables/abstract_cache.pkl`

### Step 2: Fine-tune dense retriever

Fine-tunes BGE-M3 with triplet loss (anchor=claim, positive=supporting abstract, negative=NEI abstract), then encodes the full corpus.

```bash
python scripts/train_dense_retriever.py
```

**Outputs:** `bge-m3-finetuned-triplet/` (model), `dense_embeddings_ft.pkl.gz` (embeddings)

### Step 3: Train cross-encoder reranker

Two-stage training: first with retrieval-mined negatives, then refined with the stage-1 model's hard negatives. Also generates top-10 predictions per test claim.

```bash
python scripts/train_reranker.py
```

**Outputs:** `model_stage1_FIXED/`, `model_stage2_FIXED/` (models), `predictions_FIXED.csv`

### Step 4: Classification

Three alternative classification approaches:

#### 4a. RankGPT ranking

Combines GPT-4 permutation-based ranking with cross-encoder scores via score fusion.

```bash
python scripts/run_rankgpt.py
```

#### 4b. LLM classification (GPT-4)

Four strategies available:

```bash
# 0-shot single-stage
python scripts/run_llm_classification.py --strategy 0shot-1stage --predictions-csv predictions_FIXED.csv

# 0-shot two-stage (evidentiary filter + supports/refutes)
python scripts/run_llm_classification.py --strategy 0shot-2stage --predictions-csv predictions_FIXED.csv

# Hybrid 0-shot
python scripts/run_llm_classification.py --strategy hybrid-0shot --predictions-csv predictions_FIXED.csv

# Hybrid few-shot
python scripts/run_llm_classification.py --strategy hybrid-fewshot --predictions-csv predictions_FIXED.csv
```

#### 4c. Fine-tuned DeBERTa classifier

Trains DeBERTa-v3-base-mnli with focal loss and weighted sampling, then runs inference.

```bash
# Train only
python scripts/run_ft_classification.py --train

# Predict only (using previously trained model)
python scripts/run_ft_classification.py --predict --predictions-csv predictions_FIXED.csv

# Train and predict
python scripts/run_ft_classification.py --train --predict --predictions-csv predictions_FIXED.csv
```

## Models used

| Component | Model | Source |
|-----------|-------|--------|
| Dense retriever | BGE-M3 | `BAAI/bge-m3` (fine-tuned) |
| Sparse retriever | SPLADE-v3 | `naver/splade-v3` |
| Cross-encoder reranker | MS-MARCO MiniLM | `cross-encoder/ms-marco-MiniLM-L-6-v2` (fine-tuned) |
| LLM ranker / classifier | GPT-4.1 | OpenAI API |
| Fine-tuned classifier | DeBERTa-v3-base | `MoritzLaurer/DeBERTa-v3-base-mnli` (fine-tuned) |

## Generated artifacts

| File | Description |
|------|-------------|
| `bm25_abstract_tokenized.pkl` | Tokenized abstract corpus for BM25 |
| `splade_v3_docs.npz` | SPLADE sparse document-term matrix |
| `abstract_variables/abstract_cache.pkl` | Cached abstract texts, IDs, and mapping |
| `triplets/` | Training triplets (JSONL) |
| `bge-m3-finetuned-triplet/` | Fine-tuned dense retriever checkpoint |
| `dense_embeddings_ft.pkl.gz` | Pre-computed dense embeddings |
| `model_stage1_FIXED/`, `model_stage2_FIXED/` | Reranker checkpoints |
| `deberta-claim-checker-mnli/` | Fine-tuned classifier checkpoint |
| `predictions_*.csv` | Prediction outputs |

## Key techniques

- **Reciprocal Rank Fusion (RRF)**: Combines ranked lists from BM25, SPLADE, and dense retrieval with parameter `k=60`
- **Two-stage reranker training**: Stage 1 uses retrieval-mined negatives; Stage 2 uses the stage-1 model to find harder negatives
- **Focal loss**: Handles class imbalance in DeBERTa training with inverse-frequency weighting and `gamma=2.0`
- **Weighted oversampling**: 2x dataset per epoch with class-balanced sampling
- **Partial layer freezing**: DeBERTa encoder layers 0-7 and embeddings are frozen; only layers 8-11 + pooler + classifier are trained
- **RankGPT score fusion**: `fused = alpha * norm_CE + (1-alpha) * norm_LLM` with `alpha=0.4`
