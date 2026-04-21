import importlib
import torch

models = ["bert", "deberta", "distilbert", "electra", "roberta"]

def test_models(model_name):
    module = importlib.import_module(model_name)
    model, tokenizer = module.get_model_and_tokenizer(num_labels=2)
    model.eval()

    text = "Oh great, another Monday morning."

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print(outputs.logits)         # raw scores, shape [1, 2]
    print(outputs.logits.argmax(dim=-1))  # predicted class: 0 or 1

def main():
    for name in models:
        test_models(name)

if __name__ == "__main__":
    main()