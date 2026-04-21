from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def get_model_and_tokenizer(num_labels):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
    )

    return model, tokenizer
