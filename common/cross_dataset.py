import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.dataloader import SarcasmDataset

DATASET_NAMES = [
    "csc", "csc_a", "csc_cont", "csc_a_cont",
    "isarcasmeval",
    "mustard", "mustard_cont",
    "news_headlines",
    "sarc",
    "sarcasm_v2",
]

# Datasets that share the same underlying sentences — never test across these.
CORPUS_GROUPS = [
    {"csc", "csc_a", "csc_cont", "csc_a_cont"},
    {"mustard", "mustard_cont"},
]


def same_corpus(a: str, b: str) -> bool:
    return any(a in group and b in group for group in CORPUS_GROUPS)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += len(batch["labels"])
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "/app/data"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/app/outputs"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intra_dir = os.path.join(args.output_dir, args.encoder, "intra")

    if not os.path.isdir(intra_dir):
        print(f"[{args.encoder}] no intra models found at {intra_dir}, skipping.")
        return

    train_datasets = sorted(
        d for d in os.listdir(intra_dir)
        if os.path.isdir(os.path.join(intra_dir, d))
    )

    test_paths = {
        ds: os.path.join(args.data_dir, f"{ds}_test.tsv")
        for ds in DATASET_NAMES
        if os.path.exists(os.path.join(args.data_dir, f"{ds}_test.tsv"))
    }

    for train_dataset in train_datasets:
        model_path = os.path.join(intra_dir, train_dataset)
        print(f"[{args.encoder}][{train_dataset}] loading model...")

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"  failed to load: {e}")
            continue

        model.to(device)

        results = {}
        for test_dataset, test_path in sorted(test_paths.items()):
            if same_corpus(train_dataset, test_dataset) and test_dataset != train_dataset:
                print(f"  [{test_dataset}] skipped (same source corpus as {train_dataset})")
                continue
            try:
                ds = SarcasmDataset(test_path, tokenizer)
                loader = DataLoader(ds, batch_size=args.batch_size)
                acc = evaluate(model, loader, device)
                results[test_dataset] = round(acc, 6)
                tag = " <-- intra" if test_dataset == train_dataset else ""
                print(f"  [{test_dataset}] acc={acc:.4f}{tag}")
            except Exception as e:
                print(f"  [{test_dataset}] error: {e}")
                results[test_dataset] = None

        out_dir = os.path.join(args.output_dir, args.encoder, "cross", train_dataset)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(
                {"encoder": args.encoder, "trained_on": train_dataset, "results": results},
                f, indent=2,
            )
        print(f"  saved -> {out_dir}/results.json")

        del model
        torch.cuda.empty_cache()

    print(f"[{args.encoder}] cross-dataset evaluation complete.")


if __name__ == "__main__":
    main()
