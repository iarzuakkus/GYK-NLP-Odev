from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text  # önceden yazdığımız temizleme fonksiyonu

# Etiket isimleri (GoEmotions - 28 sınıf)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# FastAPI uygulaması
app = FastAPI(title="GoEmotions LSTM API", version="1.0")

# Model ve tokenizer yükle
model = load_model("goemotions_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Giriş şeması
class TextInput(BaseModel):
    text: str
    threshold: Union[float, None] = 0.3

# Tahmin fonksiyonu
def predict_emotions(text: str, threshold: float = 0.3, max_len: int = 100):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    probs = model.predict(padded)[0]

    predictions = [
        {"label": emotion_labels[i], "score": float(round(prob, 4))}
        for i, prob in enumerate(probs) if prob > threshold
    ]
    return predictions if predictions else [{"label": "none", "score": 0.0}]


@app.post("/predict")
def get_emotions(input_data: TextInput):
    results = predict_emotions(input_data.text, threshold=input_data.threshold)
    return {"input": input_data.text, "emotions": results}


# python -m uvicorn main:app --reload