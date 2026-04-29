import pandas as pd
import random
import time
from openai import OpenAI
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

client = OpenAI(
    api_key="sk-81f248ba7b734fe1a0722be1b389ae55",
    base_url="https://llm-api.arc.vt.edu/api/v1/"
)

# MODEL = "gpt-oss-120b"
MODEL = "Kimi-K2.6"

DATASETS = ["csc", "isarcasmeval", "mustard", "news_headlines", "sarc", "sarcasm_v2"]


def load_dataset(name, split):
    path = PROJECT_ROOT / "data" / "processed" / f"{name}_{split}.tsv"
    return pd.read_csv(path, sep="\t")

def zero_shot_prompt(text):
    return f"""
You are a sarcasm detection expert.

Classify the following text as Sarcastic or Not Sarcastic.

Text:
{text}

Answer only: Sarcastic or Not Sarcastic.
"""


def build_few_shot(train_df, k=4):
    # Select balanced few-shot examples:
    # 2 sarcastic + 2 not sarcastic when k = 4

    sarcastic_examples = train_df[train_df["label"] == 1].sample(
        n=k // 2,
        random_state=5624
    )

    non_sarcastic_examples = train_df[train_df["label"] == 0].sample(
        n=k // 2,
        random_state=5624
    )

    examples = pd.concat([sarcastic_examples, non_sarcastic_examples])
    examples = examples.sample(frac=1, random_state=5805)

    prompt = "You are a sarcasm detection expert.\n\n"

    for _, row in examples.iterrows():
        label = "Sarcastic" if row["label"] == 1 else "Not Sarcastic"
        prompt += f"Text: {row['text']}\nLabel: {label}\n\n"

    return prompt


def few_shot_prompt(base_prompt, text):
    return base_prompt + f"\nText: {text}\nAnswer only: Sarcastic or Not Sarcastic."


# ===== LLM =====
def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        if response is None:
            print("Warning: API returned None")
            return "Not Sarcastic"

        if not hasattr(response, "choices") or len(response.choices) == 0:
            print("Warning: API returned no choices")
            return "Not Sarcastic"

        content = response.choices[0].message.content

        if content is None:
            print("Warning: API returned empty content")
            return "Not Sarcastic"

        return content.strip()

    except Exception as e:
        print(f"API error: {e}")
        return "Not Sarcastic"


def clean_output(output):
    output = output.lower()
    if "not sarcastic" in output:
        return 0
    elif "sarcastic" in output:
        return 1
    return 0


def evaluate(y_true, y_pred):
    return {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0)
    }

# if __name__ == "__main__":
#
#     results = []
#
#     SAMPLE_SIZE = 50
#
#     for train_ds in DATASETS:
#         train_df = load_dataset(train_ds, "train")
#
#         few_shot_base = build_few_shot(train_df)
#
#         for test_ds in DATASETS:
#             test_df = load_dataset(test_ds, "test")
#
#             y_true = []
#             y_pred_zero = []
#             y_pred_few = []
#
#             for i in range(min(SAMPLE_SIZE, len(test_df))):
#                 text = test_df.iloc[i]["text"]
#                 label = test_df.iloc[i]["label"]
#
#                 # Zero-shot
#                 z_prompt = zero_shot_prompt(text)
#                 z_out = ask_llm(z_prompt)
#                 time.sleep(2)
#                 z_pred = clean_output(z_out)
#
#                 # Few-shot
#                 f_prompt = few_shot_prompt(few_shot_base, text)
#                 f_out = ask_llm(f_prompt)
#                 time.sleep(2)
#                 f_pred = clean_output(f_out)
#
#                 y_true.append(label)
#                 y_pred_zero.append(z_pred)
#                 y_pred_few.append(f_pred)
#
#                 print(f"{train_ds}->{test_ds} | {i}")
#
#             zero_metrics = evaluate(y_true, y_pred_zero)
#             few_metrics = evaluate(y_true, y_pred_few)
#
#             results.append({
#                 "Train": train_ds,
#                 "Test": test_ds,
#                 "Method": "Zero-shot",
#                 **zero_metrics
#             })
#
#             results.append({
#                 "Train": train_ds,
#                 "Test": test_ds,
#                 "Method": "Few-shot",
#                 **few_metrics
#             })
#
#     df = pd.DataFrame(results)
#     df[["F1", "Precision", "Recall"]] = df[["F1", "Precision", "Recall"]].round(2)
#     output_path = PROJECT_ROOT / "results" / "Kimi-K2.6_results.csv"
#     df.to_csv(output_path, index=False)
if __name__ == "__main__":

    SAMPLE_SIZE = 50
    output_path = PROJECT_ROOT / "results" / f"{MODEL}_results.csv"

    # 从这里重新开始
    START_FROM = ("news_headlines", "csc")

    started = False

    # 如果旧结果文件存在，先保留 START_FROM 之前的结果
    if output_path.exists():
        existing_df = pd.read_csv(output_path)

        dataset_pairs = [(train, test) for train in DATASETS for test in DATASETS]
        start_index = dataset_pairs.index(START_FROM)

        keep_pairs = set(dataset_pairs[:start_index])

        existing_df = existing_df[
            existing_df.apply(lambda row: (row["Train"], row["Test"]) in keep_pairs, axis=1)
        ]

        existing_df.to_csv(output_path, index=False)
        print(f"Kept {len(existing_df)} old rows before {START_FROM}")
    else:
        existing_df = pd.DataFrame()

    for train_ds in DATASETS:

        train_df = load_dataset(train_ds, "train")
        few_shot_base = build_few_shot(train_df)

        for test_ds in DATASETS:

            if not started:
                if train_ds == START_FROM[0] and test_ds == START_FROM[1]:
                    started = True
                else:
                    print(f"Skipping before start point: {train_ds}->{test_ds}")
                    continue

            print(f"\nRunning {train_ds}->{test_ds}")

            test_df = load_dataset(test_ds, "test")

            y_true = []
            y_pred_zero = []
            y_pred_few = []

            for i in range(min(SAMPLE_SIZE, len(test_df))):
                text = test_df.iloc[i]["text"]
                label = test_df.iloc[i]["label"]

                z_prompt = zero_shot_prompt(text)
                z_out = ask_llm(z_prompt)
                time.sleep(2)
                z_pred = clean_output(z_out)

                f_prompt = few_shot_prompt(few_shot_base, text)
                f_out = ask_llm(f_prompt)
                time.sleep(2)
                f_pred = clean_output(f_out)

                y_true.append(label)
                y_pred_zero.append(z_pred)
                y_pred_few.append(f_pred)

                print(f"{train_ds}->{test_ds} | {i}")

            zero_metrics = evaluate(y_true, y_pred_zero)
            few_metrics = evaluate(y_true, y_pred_few)

            df_new = pd.DataFrame([
                {
                    "Train": train_ds,
                    "Test": test_ds,
                    "Method": "Zero-shot",
                    **zero_metrics
                },
                {
                    "Train": train_ds,
                    "Test": test_ds,
                    "Method": "Few-shot",
                    **few_metrics
                }
            ])

            df_new[["F1", "Precision", "Recall"]] = df_new[
                ["F1", "Precision", "Recall"]
            ].round(2)

            df_new.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)

            print(f"Saved partial result for {train_ds}->{test_ds}")