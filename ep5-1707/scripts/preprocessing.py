import os
import json
import re
import string
import numpy as np
from datasets import Dataset
from transformers import T5Tokenizer, PreTrainedTokenizer


# === 1. Temizlik Fonksiyonu ===
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# === 2. Preprocessing Fonksiyonu ===
def preprocess_function(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_input_length: int = 512,
    max_target_length: int = 128,
    use_prefix: bool = True
) -> dict:
    # === Temizleme ===
    articles = [clean_text(text) for text in examples["article"]]
    summaries = [clean_text(text) for text in examples["summary"]]

    if use_prefix:
        articles = ["summarize: " + a for a in articles]

    # === Tokenization ===
    model_inputs = tokenizer(
        articles,
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            summaries,
            max_length=max_target_length,
            padding="max_length",
            truncation=True
        )

    # Padding'i -100 yap (loss hesaplamasında yoksay)
    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    return model_inputs



# === 3. Dosya Yolları ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "cnn_dailymail_sample.json")
ENCODED_DATA_PATH = os.path.join(BASE_DIR, "outputs", "tokenized_data.npz")

# === 4. Veriyi Yükle ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
print(f"Veri yüklendi. Toplam örnek sayısı: {len(dataset)}")

# === 5. Tokenizer ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# === 6. Tokenizasyon + Temizlik ===
tokenized_dataset = dataset.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True,
    remove_columns=["article", "summary"]
)

# === 7. NumPy Formatında Kaydet ===
X = {
    "input_ids": np.array(tokenized_dataset["input_ids"]),
    "attention_mask": np.array(tokenized_dataset["attention_mask"]),
}
y = np.array(tokenized_dataset["labels"])

np.savez_compressed(ENCODED_DATA_PATH, **X, labels=y)
print(f"Tokenize veri kaydedildi → outputs/tokenized_data.npz")
