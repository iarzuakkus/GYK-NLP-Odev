import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # sayıları sil
    text = text.translate(str.maketrans('', '', string.punctuation))  # noktalama sil
    return text

def preprocess_data(csv_path, max_words=10000, max_len=100):
    # 1. Veri setini oku
    df = pd.read_csv(csv_path)

    # 2. Sadece son 28 sütunu etiket olarak al
    emotion_cols = df.columns[-28:]

    # 3. Etiket sütunlarında sadece '0' ve '1' olan satırları tut
    mask = df[emotion_cols].applymap(lambda x: str(x).strip() in ['0', '1']).all(axis=1)
    df = df[mask].copy()

    # 4. Etiketleri float32'ye çevir
    df[emotion_cols] = df[emotion_cols].astype("float32")
    labels = df[emotion_cols].values

    # 5. Text temizle
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    # 6. Eğitim-test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], labels, test_size=0.2, random_state=42
    )

    # 7. Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    # 8. Sayısallaştırma + Padding
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    # 9. Dön
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, len(tokenizer.word_index) + 1

# 🔍 Kontrol bloğu
if __name__ == "__main__":
    csv_path = "data/goemotions_merged.csv"
    X_train, X_test, y_train, y_test, tokenizer, vocab_size = preprocess_data(csv_path)

    print("✅ Preprocess tamam.")
    print("📏 X_train shape:", X_train.shape)
    print("📏 y_train shape:", y_train.shape)
    print("🔢 y_train dtype:", y_train.dtype)
    print("📚 Örnek metin:", tokenizer.sequences_to_texts([X_train[0]])[0][:100])
    print("🧠 Vocab size:", vocab_size)
