from datasets import load_dataset
import json
import os

# === Veri setini yükle ===
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]

# === İlk 1000 örneği al ===
sample_data = [
    {
        "article": example["article"],
        "summary": example["highlights"]
    }
    for example in train_data.select(range(1000))
]

# === Dosyayı kaydet ===
os.makedirs("ep5-1707/data", exist_ok=True)
with open("ep5-1707/data/cnn_dailymail_sample.json", "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=2)

print("1000 örnek başarıyla kaydedildi → ep5-1707/data/cnn_dailymail_sample.json")
