import argparse
import importlib
import json
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.dataloader import SarcasmDataset

VALID_ENCODERS = {"bert", "roberta", "deberta", "distilbert", "electra"}
DATASET_NAMES = {
    "csc",
    "isarcasmeval",
    "mustard",
    "news_headlines",
    "sarc",
    "sarcasm_v2",
}
OUTPUT_DIR = "/app/outputs"
NUM_LABELS = 2

ENCODER_DEFAULTS = {
    "bert": {"lr": 2e-5, "batch_size": 16, "epochs": 3},
    "roberta": {"lr": 3e-5, "batch_size": 16, "epochs": 3},
    "distilbert": {"lr": 3e-5, "batch_size": 32, "epochs": 2},
    "deberta": {"lr": 1e-5, "batch_size": 8, "epochs": 4},
    "electra": {"lr": 2e-5, "batch_size": 16, "epochs": 3},
}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.encoder}] using device: {device}")

    encoder_module = importlib.import_module(args.encoder)
    data_dir = args.data_dir

    for dataset in sorted(DATASET_NAMES):
        train_path = os.path.join(data_dir, f"{dataset}_train.tsv")
        val_path = os.path.join(data_dir, f"{dataset}_val.tsv")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"[{args.encoder}][{dataset}] skipping — data files not found")
            continue

        print(f"[{args.encoder}][{dataset}] loading fresh model and tokenizer...")
        model, tokenizer = encoder_module.get_model_and_tokenizer(NUM_LABELS)
        model.to(device)

        print(f"[{args.encoder}][{dataset}] loading data...")
        train_dataset = SarcasmDataset(train_path, tokenizer)
        val_dataset = SarcasmDataset(val_path, tokenizer)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        print(
            f"[{args.encoder}][{dataset}] {len(train_dataset)} train, {len(val_dataset)} val"
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        best_val_acc = 0.0
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            print(
                f"[{args.encoder}][{dataset}] epoch {epoch}/{args.epochs} starting..."
            )
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += outputs.loss.item()

            avg_loss = total_loss / len(train_loader)
            val_acc = evaluate(model, val_loader, device)
            history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})
            print(
                f"[{args.encoder}][{dataset}] epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save(model, tokenizer, args.encoder, dataset, history)

        print(f"[{args.encoder}][{dataset}] done. best_val_acc={best_val_acc:.4f}")

        # free GPU memory before next dataset
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    print(f"[{args.encoder}] all datasets complete.")


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


def save(model, tokenizer, encoder_name, dataset_name, history):
    out = os.path.join(OUTPUT_DIR, encoder_name, "intra", dataset_name)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=VALID_ENCODERS)
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "/app/data"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "3")))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "16"))
    )
    parser.add_argument("--lr", type=float, default=float(os.getenv("LR", "2e-5")))
    args = parser.parse_args()
    train(args)
