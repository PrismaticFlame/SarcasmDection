"""
Normalize each raw dataset into a two-column TSV: text <tab> label (0 or 1).
Output files land in data/processed/<dataset>_<split>.tsv

Run from the project root:  python data/preprocessing.py
"""

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
    # Scores range ~1–9; treat >=5 as sarcastic (majority evaluator agreement)
    df["label"] = (df["sarcasm_score_by_evaluator"] >= 5).astype(int)
    df = df[["response_text", "label"]].rename(columns={"response_text": "text"})
    return df


def process_mustard() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "mustard", "sarcasm_data.json")
    data = json.load(open(path))
    rows = [
        {"text": entry["utterance"], "label": int(entry["sarcasm"])}
        for entry in data.values()
    ]
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


# ---------------------------------------------------------------------------
# Split and write
# ---------------------------------------------------------------------------

PROCESSORS = {
    "news_headlines": process_news_headlines,
    "isarcasmeval":   process_isarcasmeval,
    "csc":            process_csc,
    "mustard":        process_mustard,
    "sarcasm_v2":     process_sarcasm_v2,
}


def split_and_save(name: str, df: pd.DataFrame):
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    train, temp = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"])
    val, test  = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp["label"])

    for split, data in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(OUT_DIR, f"{name}_{split}.tsv")
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


if __name__ == "__main__":
    main()
