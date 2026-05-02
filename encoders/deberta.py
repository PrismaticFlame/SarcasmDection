import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer


def get_model_and_tokenizer(num_labels):
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )
    return model, tokenizer
