from transformers import RobertaTokenizerFast, RobertaForSequenceClassification


def get_model_and_tokenizer(num_labels):
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )

    return model, tokenizer
