#!/usr/bin/env python3
"""GPT-4 based claim classification with multiple strategies.

Replaces: LLM_classification.ipynb

Requires OPENAI_API_KEY environment variable to be set.

Usage:
    python scripts/run_llm_classification.py --strategy 0shot-1stage
    python scripts/run_llm_classification.py --strategy 0shot-2stage
    python scripts/run_llm_classification.py --strategy hybrid-0shot
    python scripts/run_llm_classification.py --strategy hybrid-fewshot
"""

import argparse
import os

import pandas as pd

from climatecheck.data import load_claims_test, load_abstracts
from climatecheck.llm_classification import (
    classify_0shot_1stage,
    classify_0shot_2stage,
    classify_hybrid,
)


def main():
    parser = argparse.ArgumentParser(description="LLM-based claim classification")
    parser.add_argument(
        "--strategy",
        choices=["0shot-1stage", "0shot-2stage", "hybrid-0shot", "hybrid-fewshot"],
        default="0shot-1stage",
        help="Classification strategy to use",
    )
    parser.add_argument(
        "--predictions-csv",
        default="predictions.csv",
        help="Input CSV with claim_id, abstract_id columns",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (auto-generated if not specified)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable")

    # Load data
    verification_dataset = load_claims_test()
    abstracts_dataset = load_abstracts()

    claim_map = {entry["claim_id"]: entry["claim"] for entry in verification_dataset}
    abstract_map = {entry["abstract_id"]: entry["abstract"] for entry in abstracts_dataset}

    df = pd.read_csv(args.predictions_csv)
    df["claim"] = df["claim_id"].map(claim_map)
    df["abstract"] = df["abstract_id"].map(abstract_map)

    # Classify
    print(f"=== Running strategy: {args.strategy} ===")
    if args.strategy == "0shot-1stage":
        df = classify_0shot_1stage(df, api_key)
        default_output = "predictions_0shot_1stage.csv"
    elif args.strategy == "0shot-2stage":
        df = classify_0shot_2stage(df, api_key)
        default_output = "predictions_0shot_2stage.csv"
    elif args.strategy == "hybrid-0shot":
        df = classify_hybrid(df, api_key, use_fewshot=False)
        default_output = "predictions_hybrid_0shot.csv"
    elif args.strategy == "hybrid-fewshot":
        df = classify_hybrid(df, api_key, use_fewshot=True)
        default_output = "predictions_hybrid_fewshot.csv"

    output_path = args.output_csv or default_output
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
