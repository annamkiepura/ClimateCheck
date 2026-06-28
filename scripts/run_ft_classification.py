#!/usr/bin/env python3
"""Fine-tune DeBERTa-v3 classifier and run inference.

Replaces: ft_classification.ipynb

Usage:
    python scripts/run_ft_classification.py --train
    python scripts/run_ft_classification.py --predict --predictions-csv predictions.csv
    python scripts/run_ft_classification.py --train --predict --predictions-csv predictions.csv
"""

import argparse

import pandas as pd

from climatecheck.data import load_claims_train, load_claims_test, load_abstracts
from climatecheck.ft_classification import train_classifier, predict_with_classifier


def main():
    parser = argparse.ArgumentParser(description="Fine-tuned DeBERTa classification")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument("--predict", action="store_true", help="Run predictions")
    parser.add_argument(
        "--predictions-csv",
        default="predictions.csv",
        help="Input CSV with claim_id, abstract_id columns (for --predict)",
    )
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.error("Specify at least one of --train or --predict")

    if args.train:
        print("=== Training classifier ===")
        claims_dataset = load_claims_train()
        train_classifier(claims_dataset)

    if args.predict:
        print("\n=== Running predictions ===")
        abstracts_dataset = load_abstracts()
        verification_dataset = load_claims_test()
        test_df = pd.read_csv(args.predictions_csv)
        predict_with_classifier(test_df, abstracts_dataset, verification_dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
