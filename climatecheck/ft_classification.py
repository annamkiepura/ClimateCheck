"""Fine-tune DeBERTa-v3 for climate claim classification with focal loss."""

import os
import random
from collections import Counter

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from .config import DEBERTA_MODEL, CLASSIFIER_DIR, SEED

# ── Reproducibility ─────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Focal loss ──────────────────────────────────────────────────────────────

def focal_loss(logits, labels, alpha, gamma=2.0):
    ce = F.cross_entropy(logits, labels, reduction="none", weight=alpha)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ── Training ────────────────────────────────────────────────────────────────

def train_classifier(claims_dataset=None):
    """
    Fine-tune DeBERTa-v3-base-mnli for 3-class claim verification.

    Parameters
    ----------
    claims_dataset : HF Dataset or None
        If None, loads from the HuggingFace Hub.
    """
    if claims_dataset is None:
        from .data import load_claims_train
        claims_dataset = load_claims_train()

    full_ds = claims_dataset

    # Stratified split
    idx = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.10,
        random_state=SEED,
        stratify=full_ds["annotation"],
    )
    ds = DatasetDict({
        "train": full_ds.select(train_idx),
        "validation": full_ds.select(val_idx),
    })

    # Label mapping
    unique_labels = sorted(set(ds["train"]["annotation"]))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    # Tokenizer & preprocessing
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL)

    def preprocess(ex):
        enc_claim = tokenizer(ex["claim"], truncation=True, max_length=64)
        enc_abs = tokenizer(ex["abstract"], truncation=True, max_length=448)
        enc = tokenizer.pad(
            {k: enc_claim[k] + enc_abs[k][1:] for k in enc_claim},
            max_length=512,
        )
        enc["labels"] = label2id[ex["annotation"]]
        return enc

    ds_enc = ds.map(preprocess, batched=False, remove_columns=ds["train"].column_names)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Model with frozen layers 0-7
    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_MODEL,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
    )

    for n, p in model.named_parameters():
        if "encoder.layer." in n:
            layer_num = int(n.split("encoder.layer.")[1].split(".")[0])
            if layer_num < 8:
                p.requires_grad = False
        elif n.startswith("embeddings."):
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.1f} M")

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # Class weights for focal loss
    counts = Counter(ds["train"]["annotation"])
    alpha = torch.tensor([1 / counts[lbl] for lbl in unique_labels])
    alpha = alpha / alpha.sum() * len(unique_labels)

    sample_weights = [alpha[label2id[lbl]].item() for lbl in ds["train"]["annotation"]]

    # Training arguments
    args = TrainingArguments(
        CLASSIFIER_DIR,
        bf16=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=10,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
    )

    # Custom trainer with focal loss + oversampling
    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = focal_loss(outputs.logits, labels, alpha.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights) * 2,
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    trainer = FocalTrainer(
        model=model,
        args=args,
        train_dataset=ds_enc["train"],
        eval_dataset=ds_enc["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(CLASSIFIER_DIR)
    tokenizer.save_pretrained(CLASSIFIER_DIR)
    print(f"Model saved to {CLASSIFIER_DIR}")
    return model, tokenizer, id2label


# ── Inference ───────────────────────────────────────────────────────────────

def predict_with_classifier(test_df, abstracts_dataset, verification_dataset,
                            model_dir=None):
    """
    Run inference on test predictions using the fine-tuned DeBERTa classifier.

    Parameters
    ----------
    test_df : pd.DataFrame
        Must have columns: claim_id, abstract_id.
    abstracts_dataset : HF Dataset
        The abstracts corpus.
    verification_dataset : HF Dataset
        The verification (test) claims.
    model_dir : str or None
        Path to the fine-tuned model directory.
    """
    import pandas as pd

    model_dir = model_dir or CLASSIFIER_DIR

    # Map ids to texts
    claim_map = {entry["claim_id"]: entry["claim"] for entry in verification_dataset}
    abstract_map = {entry["abstract_id"]: entry["abstract"] for entry in abstracts_dataset}

    test_df["claim"] = test_df["claim_id"].map(claim_map)
    test_df["abstract"] = test_df["abstract_id"].map(abstract_map)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # Tokenize
    ds_test = Dataset.from_pandas(test_df, preserve_index=True)

    def preprocess(batch):
        return tokenizer(batch["claim"], batch["abstract"], truncation=True, max_length=512)

    ds_enc = ds_test.map(preprocess, batched=True, remove_columns=ds_test.column_names)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(ds_enc, batch_size=32, collate_fn=collator)

    # Predict
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to("cuda") for k, v in batch.items()}
            logits = model(**batch_gpu).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.append(preds)

    pred_ids = np.concatenate(predictions)
    pred_labels = [id2label[i] for i in pred_ids]

    test_df["label"] = pred_labels
    test_df.to_csv("predictions_deberta.csv", index=False)
    print("Saved predictions to predictions_deberta.csv")
    return test_df
