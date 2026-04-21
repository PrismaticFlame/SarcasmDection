from transformers import BertForSequenceClassification, BertTokenizerFast


def get_model_and_tokenizer(num_labels):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
    )

    return model, tokenizer



