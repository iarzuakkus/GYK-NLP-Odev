import os
import json
import numpy as np
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from datasets import Dataset

# === 1. Dosya Yolları ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENCODED_DATA_PATH = os.path.join(BASE_DIR, "outputs", "tokenized_data.npz")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "t5-small-summary")
HYPERPARAM_PATH = os.path.join(BASE_DIR, "outputs", "hyperparameters.json")
TRAIN_LOG_PATH = os.path.join(BASE_DIR, "outputs", "train_log.txt")

# === 2. Veriyi Yükle ===
data = np.load(ENCODED_DATA_PATH)
X_input_ids = data["input_ids"]
X_attention_mask = data["attention_mask"]
y_labels = data["labels"]

# === 3. Dataset Oluştur ===
train_dataset = Dataset.from_dict({
    "input_ids": X_input_ids,
    "attention_mask": X_attention_mask,
    "labels": y_labels
})
print(f"Eğitim verisi hazırlandı. Örnek sayısı: {len(train_dataset)}")

# === 4. Model ve Tokenizer ===
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# === 5. Eğitim Parametreleri ===
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    per_device_train_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="no",
    save_strategy="no",
    logging_steps=10,
    report_to="none",
    disable_tqdm=False
)

# === 6. Trainer Nesnesi ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === 7. Eğitimi Başlat ===
print("Eğitim başlıyor...")
train_result = trainer.train()
print("Eğitim tamamlandı.")

# === 8. Model ve Tokenizer Kaydet ===
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model kaydedildi → {MODEL_SAVE_PATH}")

# === 9. Hiperparametre Kaydet ===
hyperparams = {
    "model_name": model_name,
    "num_train_epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "max_input_length": 512,
    "max_target_length": 128
}
with open(HYPERPARAM_PATH, "w", encoding="utf-8") as f:
    json.dump(hyperparams, f, indent=2)
print(f"Hiperparametreler kaydedildi → {HYPERPARAM_PATH}")

# === 10. Eğitim Loglarını Kaydet ===
log_history = trainer.state.log_history
with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as f:
    for log in log_history:
        f.write(json.dumps(log) + "\n")
print(f"Eğitim logları kaydedildi → {TRAIN_LOG_PATH}")
