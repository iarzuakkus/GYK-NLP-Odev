import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text  # temizleme fonksiyonu
import pandas as pd

# 1. Etiket adlarını tanımla (goemotions 28 etiket)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# 2. Model ve tokenizer yükle
model = load_model("goemotions_lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 3. Tahmin fonksiyonu
def predict_emotions(text, max_len=100, threshold=0.3):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    probs = model.predict(padded)[0]

    # threshold üstündeki etiketleri al
    predicted = [(emotion_labels[i], round(prob, 3)) for i, prob in enumerate(probs) if prob > threshold]

    return predicted if predicted else ["Hiçbir duygu eşiği geçemedi."]

# 4. Kullanıcıdan metin al
if __name__ == "__main__":
    print("🎤 Bir cümle gir ve duygu tahmini al!")
    while True:
        text = input("\nMetin (çıkmak için q): ")
        if text.lower() in ['q', 'quit', 'exit']:
            break

        results = predict_emotions(text)
        print("\n🎯 Tahmin Edilen Duygular:")
        if isinstance(results, list):
            for item in results:
                print(f" - {item[0]} ({item[1]*100:.1f}%)" if isinstance(item, tuple) else f" - {item}")
        else:
            print(results)
