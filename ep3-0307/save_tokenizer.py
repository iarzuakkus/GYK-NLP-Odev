import pickle
from preprocess import preprocess_data  # tokenizer zaten burada oluşuyor

if __name__ == "__main__":
    csv_path = "data/goemotions_merged.csv"

    # Sadece tokenizer'ı almak için çalıştırıyoruz
    _, _, _, _, tokenizer, _ = preprocess_data(csv_path)

    # Kaydet
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Tokenizer başarıyla kaydedildi: tokenizer.pkl")
