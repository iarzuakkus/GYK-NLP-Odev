from datasets import load_dataset
import json
import os

def select_and_save_data(
    line_range: tuple,
    file_path: str,
    append: bool = False,
    split: str = "train"
):
    """
    Selects a range of examples from the specified CNN/DailyMail dataset split and saves them to a JSON file.

    Args:
        line_range (tuple): (start, end) index range to select
        file_path (str): Path to save the output JSON
        append (bool): If True, appends to existing data in the file
        split (str): Dataset split to use, e.g., 'train', 'test', or 'validation'
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    data_split = dataset[split]

    data = [
        {
            "article": example["article"],
            "summary": example["highlights"]
        }
        for example in data_split.select(range(*line_range))
    ]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if append and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        data = existing_data + data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {split} examples from {line_range[0]} to {line_range[1]-1} → {file_path}")

if __name__ == "__main__":
    output_file = "ep5-1707/data/cnn_dailymail_validation.json"
    #select_and_save_data((0, 200), output_file, append=False, split="test")  # 200 test samples
    select_and_save_data((0, 200), output_file, append=True, split="validation")  # 200 validation samples
