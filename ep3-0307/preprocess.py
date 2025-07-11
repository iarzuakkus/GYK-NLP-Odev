import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Etiket kolonları (son 28 sütun)
    emotion_cols = df.columns[-28:]

    # Etiketleri doğrudan float'a çevir
    df[emotion_cols] = df[emotion_cols].astype("float32")
    labels = df[emotion_cols].values

    # Metinleri al
    texts = df['text'].astype(str).tolist()

    return texts, labels

def preprocess_data(texts, labels, max_words=10000, max_len=100):
    cleaned_texts = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    vocab_size = len(tokenizer.word_index) + 1

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, vocab_size

# 🔍 Kontrol bloğu
if __name__ == "__main__":
    csv_path = "data/augmented_dataset.csv"

    texts, labels = load_data(csv_path)
    X_train, X_test, y_train, y_test, tokenizer, vocab_size = preprocess_data(texts, labels)

    print("✅ Preprocess tamam.")
    print("📏 X_train shape:", X_train.shape)
    print("📏 y_train shape:", y_train.shape)
    print("🔢 y_train dtype:", y_train.dtype)
    print("📚 Örnek metin:", tokenizer.sequences_to_texts([X_train[0]])[0][:100])
    print("🧠 Vocab size:", vocab_size)
