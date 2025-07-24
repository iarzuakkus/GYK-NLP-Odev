from datasets import load_dataset
import json
import os

def select_and_save_data(line_range, file_path, split="train"):
    dataset = load_dataset("cnn_dailymail", "3.0.0")[split]
    data = [
        {"article": d["article"], "summary": d["highlights"]}
        for d in dataset.select(range(*line_range))
    ]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {split} examples from {line_range[0]} to {line_range[1]-1} → {file_path}")

if __name__ == "__main__":
    output_path = "data/cnn_dailymail_sample.json"
    select_and_save_data((0, 10000), output_path, split="train")