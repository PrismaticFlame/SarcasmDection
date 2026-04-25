import torch
from torch.utils.data import Dataset


class SarcasmDataset(Dataset):
    """
    Expects a tab-separated file with columns: text, label (0 or 1).
    Produced by data/preprocessing.py from each raw dataset.
    """

    def __init__(self, path, tokenizer, max_length=128):
        import pandas as pd

        df = pd.read_csv(path, sep="\t", dtype={"text": str})
        df = df.dropna(subset=["text"])
        self.labels = df["label"].tolist()
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
