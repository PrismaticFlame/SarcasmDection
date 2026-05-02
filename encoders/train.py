import argparse
import importlib
import json
import logging
import os
import sys
import time

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler, autocast
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
    "csc", "csc_a", "csc_cont", "csc_a_cont",
    "isarcasmeval",
    "mustard", "mustard_cont",
    "news_headlines",
    "sarc",
    "sarcasm_v2",
}
OUTPUT_DIR = "/app/outputs"
NUM_LABELS = 2

ENCODER_DEFAULTS = {
    "bert":       {"lr": 2e-5, "batch_size": 16, "epochs": 3, "warmup_ratio": 0.10, "eps": 1e-8, "clip": 1.0},
    "roberta":    {"lr": 3e-5, "batch_size": 16, "epochs": 3, "warmup_ratio": 0.10, "eps": 1e-8, "clip": 1.0},
    "distilbert": {"lr": 3e-5, "batch_size": 32, "epochs": 2, "warmup_ratio": 0.10, "eps": 1e-8, "clip": 1.0},
    "deberta":    {"lr": 5e-6, "batch_size": 8,  "epochs": 4, "warmup_ratio": 0.20, "eps": 1e-6, "clip": 1.0},
    "electra":    {"lr": 2e-5, "batch_size": 16, "epochs": 3, "warmup_ratio": 0.10, "eps": 1e-8, "clip": 1.0},
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )
        scaler = GradScaler("cuda") if device.type == "cuda" else None

        best_val_f1 = 0.0
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            nan_batches = 0
            print(
                f"[{args.encoder}][{dataset}] epoch {epoch}/{args.epochs} starting..."
            )
            for batch in train_loader:
                optimizer.zero_grad()
                with autocast("cuda", enabled=(scaler is not None)):
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                loss = outputs.loss
                if torch.isnan(loss):
                    nan_batches += 1
                    continue
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            if nan_batches:
                print(f"[{args.encoder}][{dataset}]   warning: {nan_batches} NaN batches skipped")

            avg_loss = total_loss / len(train_loader)
            metrics = evaluate(model, val_loader, device)
            history.append({"epoch": epoch, "loss": avg_loss, **metrics})
            print(
                f"[{args.encoder}][{dataset}] epoch {epoch}/{args.epochs} "
                f"loss={avg_loss:.4f}  acc={metrics['accuracy']:.4f}  "
                f"p={metrics['precision']:.4f}  r={metrics['recall']:.4f}  "
                f"f1={metrics['f1']:.4f}"
            )

            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                save(model, tokenizer, args.encoder, dataset, history)

        print(f"[{args.encoder}][{dataset}] done. best_val_f1={best_val_f1:.4f}")

        # free GPU memory before next dataset
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    print(f"[{args.encoder}] all datasets complete.")


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return {
        "accuracy":  round(float(accuracy_score(all_labels, all_preds)), 6),
        "precision": round(float(precision_score(all_labels, all_preds, zero_division=0)), 6),
        "recall":    round(float(recall_score(all_labels, all_preds, zero_division=0)), 6),
        "f1":        round(float(f1_score(all_labels, all_preds, zero_division=0)), 6),
    }


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
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--eps",          type=float, default=None)
    parser.add_argument("--clip",         type=float, default=None)
    args = parser.parse_args()

    # Fill any unset args from per-encoder defaults
    defaults = ENCODER_DEFAULTS[args.encoder]
    if args.epochs       is None: args.epochs       = int(os.getenv("EPOCHS",       defaults["epochs"]))
    if args.batch_size   is None: args.batch_size   = int(os.getenv("BATCH_SIZE",   defaults["batch_size"]))
    if args.lr           is None: args.lr           = float(os.getenv("LR",         defaults["lr"]))
    if args.warmup_ratio is None: args.warmup_ratio = float(os.getenv("WARMUP_RATIO", defaults["warmup_ratio"]))
    if args.eps          is None: args.eps          = float(os.getenv("EPS",         defaults["eps"]))
    if args.clip         is None: args.clip         = float(os.getenv("CLIP",        defaults["clip"]))

    train(args)
