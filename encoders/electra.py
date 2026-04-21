from transformers import BertTokenizer, ElectraForSequenceClassification


def get_model_and_tokenizer(num_labels):
    tokenizer = BertTokenizer.from_pretrained("google/electra-base-discriminator")
    model = ElectraForSequenceClassification.from_pretrained(
        "google/electra-base-discriminator",
        num_labels=num_labels,
    )

    return model, tokenizer
