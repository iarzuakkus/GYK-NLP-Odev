from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration

# === Dosya yolu çözümünü her yerden çalışır hale getir ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "t5-small-summary"

# === Tokenizer ve Modeli Yükle (yalnızca local) ===
tokenizer = T5Tokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH), local_files_only=True)

def predict_summary(text: str, max_length: int = 128) -> str:
    input_text = "summarize: " + text.lower().strip()
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === CLI'den test etmek istersen ===
if __name__ == "__main__":
    sample = (
        "In recent years, the use of artificial intelligence has expanded rapidly across various industries. "
        "From healthcare and finance to transportation and education, AI technologies are transforming the way we live and work. "
        "While many experts praise the potential of AI to improve efficiency and decision-making, concerns about privacy, bias, and job displacement remain. "
        "Governments and organizations are now working together to develop ethical guidelines and regulations to ensure AI is used responsibly and fairly."
    )
    print("Özet:", predict_summary(sample))
