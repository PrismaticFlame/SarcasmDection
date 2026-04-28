"""
Re-evaluate all saved intra models with full metrics (accuracy, precision, recall, F1)
without retraining. Saves intra_metrics.json alongside each model's history.json.

Run from the project root (host, no Docker needed):
    python encoders/reeval.py

Or inside Docker via the orchestrator (SCRIPT=reeval.py).
"""
import argparse
import json
import os
import sys

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.dataloader import SarcasmDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    return {
        "accuracy":  round(float(accuracy_score(all_labels, all_preds)), 6),
        "precision": round(float(precision_score(all_labels, all_preds, zero_division=0)), 6),
        "recall":    round(float(recall_score(all_labels, all_preds, zero_division=0)), 6),
        "f1":        round(float(f1_score(all_labels, all_preds, zero_division=0)), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "encoders", "outputs")),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data", "processed")),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for encoder in sorted(os.listdir(args.output_dir)):
        intra_dir = os.path.join(args.output_dir, encoder, "intra")
        if not os.path.isdir(intra_dir):
            continue

        for dataset in sorted(os.listdir(intra_dir)):
            model_path = os.path.join(intra_dir, dataset)
            if not os.path.isdir(model_path):
                continue

            val_path  = os.path.join(args.data_dir, f"{dataset}_val.tsv")
            test_path = os.path.join(args.data_dir, f"{dataset}_test.tsv")

            if not os.path.exists(val_path) or not os.path.exists(test_path):
                print(f"[{encoder}][{dataset}] skipping — data files not found")
                continue

            print(f"[{encoder}][{dataset}] loading model...")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                print(f"  failed to load: {e}")
                continue

            model.to(device)

            val_metrics  = evaluate(model, DataLoader(SarcasmDataset(val_path,  tokenizer), batch_size=args.batch_size), device)
            test_metrics = evaluate(model, DataLoader(SarcasmDataset(test_path, tokenizer), batch_size=args.batch_size), device)

            print(
                f"  val  — acc={val_metrics['accuracy']:.4f}  p={val_metrics['precision']:.4f}"
                f"  r={val_metrics['recall']:.4f}  f1={val_metrics['f1']:.4f}"
            )
            print(
                f"  test — acc={test_metrics['accuracy']:.4f}  p={test_metrics['precision']:.4f}"
                f"  r={test_metrics['recall']:.4f}  f1={test_metrics['f1']:.4f}"
            )

            out = {"encoder": encoder, "dataset": dataset, "val": val_metrics, "test": test_metrics}
            with open(os.path.join(model_path, "intra_metrics.json"), "w") as f:
                json.dump(out, f, indent=2)

            del model
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
