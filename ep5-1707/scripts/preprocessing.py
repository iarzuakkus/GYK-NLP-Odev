import os
import json
import re
import string
import numpy as np
from datasets import Dataset
from transformers import T5Tokenizer, PreTrainedTokenizer


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_function(examples: dict, tokenizer: PreTrainedTokenizer,
                        max_input_length: int = 512, max_target_length: int = 128,
                        use_prefix: bool = True) -> dict:
    articles = [clean_text(text) for text in examples["article"]]
    summaries = [clean_text(text) for text in examples["summary"]]

    if use_prefix:
        articles = ["summarize: " + a for a in articles]

    model_inputs = tokenizer(articles, max_length=max_input_length,
                             padding="max_length", truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summaries, max_length=max_target_length,
                           padding="max_length", truncation=True)

    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    return model_inputs


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZED_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(TOKENIZED_DIR, exist_ok=True)

data_files = {
    "train": os.path.join(BASE_DIR, "data", "cnn_dailymail_sample.json"),
    "validation": os.path.join(BASE_DIR, "data", "cnn_dailymail_validation.json")
}

tokenizer = T5Tokenizer.from_pretrained("t5-small")

for split, path in data_files.items():
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = Dataset.from_list(raw_data)
    print(f"{split.capitalize()} verisi yüklendi. Örnek sayısı: {len(dataset)}")

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["article", "summary"]
    )

    X = {
        "input_ids": np.array(tokenized_dataset["input_ids"]),
        "attention_mask": np.array(tokenized_dataset["attention_mask"]),
    }
    y = np.array(tokenized_dataset["labels"])

    save_path = os.path.join(TOKENIZED_DIR, f"tokenized_{split}_data.npz")
    np.savez_compressed(save_path, **X, labels=y)
    print(f"{split.capitalize()} verisi kaydedildi → {save_path}")