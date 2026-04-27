"""
Normalize each raw dataset into a two-column TSV: text <tab> label (0 or 1).
Output files land in data/processed/<dataset>_<split>.tsv

Run from the project root:  python data/preprocessing.py
"""

import csv
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Per-dataset parsers — each returns a DataFrame with columns: text, label
# ---------------------------------------------------------------------------


def process_news_headlines() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "news_headlines", "Sarcasm_Headlines_Dataset.json")
    records = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(records)[["headline", "is_sarcastic"]]
    df.columns = ["text", "label"]
    return df


def process_isarcasmeval() -> pd.DataFrame:
    # Only train split has labels — test file (task_A_En_test.csv) does not
    path = os.path.join(RAW_DIR, "isarcasmeval", "train.En.csv")
    df = pd.read_csv(path)[["tweet", "sarcastic"]]
    df.columns = ["text", "label"]
    return df


def process_csc() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "csc", "data_full.csv")
    df = pd.read_csv(path)[["response_text", "sarcasm_score_by_evaluator"]]
    # Scores range 1–6; treat >=5 as sarcastic (strong evaluator agreement)
    df["label"] = (df["sarcasm_score_by_evaluator"] >= 5).astype(int)
    df = df[["response_text", "label"]].rename(columns={"response_text": "text"})
    return df


def process_csc_a() -> pd.DataFrame:
    """CSC with author (speaker) label instead of third-party evaluator."""
    path = os.path.join(RAW_DIR, "csc", "data_full.csv")
    df = pd.read_csv(path)[["response_text", "sarcasm_score_by_speaker"]]
    df["label"] = (df["sarcasm_score_by_speaker"] >= 5).astype(int)
    return df[["response_text", "label"]].rename(columns={"response_text": "text"})


def process_csc_cont() -> pd.DataFrame:
    """CSC with third-party label and situation context prepended."""
    path = os.path.join(RAW_DIR, "csc", "data_full.csv")
    df = pd.read_csv(path)[["context_text", "response_text", "sarcasm_score_by_evaluator"]]
    df["text"] = df["context_text"].str.strip() + " | " + df["response_text"].str.strip()
    df["label"] = (df["sarcasm_score_by_evaluator"] >= 5).astype(int)
    return df[["text", "label"]]


def process_csc_a_cont() -> pd.DataFrame:
    """CSC with author label and situation context prepended."""
    path = os.path.join(RAW_DIR, "csc", "data_full.csv")
    df = pd.read_csv(path)[["context_text", "response_text", "sarcasm_score_by_speaker"]]
    df["text"] = df["context_text"].str.strip() + " | " + df["response_text"].str.strip()
    df["label"] = (df["sarcasm_score_by_speaker"] >= 5).astype(int)
    return df[["text", "label"]]


def process_mustard() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "mustard", "sarcasm_data.json")
    data = json.load(open(path))
    rows = [
        {"text": entry["utterance"], "label": int(entry["sarcasm"])}
        for entry in data.values()
    ]
    return pd.DataFrame(rows)


def process_mustard_cont() -> pd.DataFrame:
    """MUStARD with prior dialogue turns prepended as context."""
    path = os.path.join(RAW_DIR, "mustard", "sarcasm_data.json")
    data = json.load(open(path))
    rows = []
    for entry in data.values():
        context = " | ".join(entry["context"]) if entry["context"] else ""
        text = context + " | " + entry["utterance"] if context else entry["utterance"]
        rows.append({"text": text, "label": int(entry["sarcasm"])})
    return pd.DataFrame(rows)


def process_sarcasm_v2() -> pd.DataFrame:
    base = os.path.join(RAW_DIR, "sarcasm_v2", "sarcasm_v2")
    files = ["GEN-sarc-notsarc.csv", "HYP-sarc-notsarc.csv", "RQ-sarc-notsarc.csv"]
    parts = []
    for f in files:
        df = pd.read_csv(os.path.join(base, f))[["class", "text"]]
        df["label"] = (df["class"] == "sarc").astype(int)
        parts.append(df[["text", "label"]])
    return pd.concat(parts, ignore_index=True)


def _parse_sarc_split(path: str, comments: dict) -> pd.DataFrame:
    """Parse a SARC balanced CSV into (text, label) rows.

    Each line is: ancestor_id|resp_id1 resp_id2|label1 label2
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for parts in reader:
            if len(parts) < 3:
                continue
            resp_ids = parts[1].strip().split()
            labels = parts[2].strip().split()
            for rid, lab in zip(resp_ids, labels):
                if rid in comments:
                    rows.append({"text": comments[rid]["text"], "label": int(lab)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split and write
# ---------------------------------------------------------------------------

PROCESSORS = {
    "news_headlines": process_news_headlines,
    "isarcasmeval": process_isarcasmeval,
    "csc": process_csc,
    "csc_a": process_csc_a,
    "csc_cont": process_csc_cont,
    "csc_a_cont": process_csc_a_cont,
    "mustard": process_mustard,
    "mustard_cont": process_mustard_cont,
    "sarcasm_v2": process_sarcasm_v2,
}


def split_and_save(name: str, df: pd.DataFrame):
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    train, temp = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp["label"]
    )

    for split, data in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(OUT_DIR, f"{name}_{split}.tsv")
        data.to_csv(out, sep="\t", index=False)
        print(f"  {split}: {len(data)} rows -> {out}")


def process_sarc():
    """SARC has its own train/test splits, so we handle it separately."""
    sarc_dir = os.path.join(RAW_DIR, "sarc")
    comments_path = os.path.join(sarc_dir, "comments.json")

    print("[sarc]")
    print("  loading comments.json ...")
    with open(comments_path) as f:
        comments = json.load(f)

    train_df = _parse_sarc_split(os.path.join(sarc_dir, "train-balanced.csv"), comments)
    test_df = _parse_sarc_split(os.path.join(sarc_dir, "test-balanced.csv"), comments)

    # Carve a validation set from training data
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=RANDOM_SEED, stratify=train_df["label"]
    )

    for split, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        data = data.dropna(subset=["text", "label"])
        data["text"] = data["text"].astype(str).str.strip()
        data = data[data["text"] != ""]
        out = os.path.join(OUT_DIR, f"sarc_{split}.tsv")
        data.to_csv(out, sep="\t", index=False)
        print(f"  {split}: {len(data)} rows -> {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, fn in PROCESSORS.items():
        print(f"[{name}]")
        try:
            df = fn()
            split_and_save(name, df)
        except FileNotFoundError as e:
            print(f"  [skip] missing file: {e}")

    try:
        process_sarc()
    except FileNotFoundError as e:
        print(f"  [skip] missing file: {e}")


if __name__ == "__main__":
    main()
