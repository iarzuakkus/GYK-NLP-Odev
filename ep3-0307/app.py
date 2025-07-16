import gradio as gr
import tensorflow as tf
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

# Model ve tokenizer yolları (Hugging Face için göreceli yol)
model = tf.keras.models.load_model("models/goemotions_lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # Eğitimde ne kullandıysan onu yaz

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded)[0]
    predicted_label_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label_map = {
        0: "Mutluluk", 1: "Üzgünlük", 2: "Öfke", 3: "Korku",
        4: "Şaşkınlık", 5: "Nefret"
        # Genişletilebilir: 27 sınıf varsa buraya eklersin
    }

    label = label_map.get(predicted_label_index, "Bilinmeyen")
    return f"{label} ({confidence:.2f} güven)"

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Bir metin girin..."),
    outputs="text",
    title="GoEmotions Duygu Analizi",
    description="Bu model, LSTM ile metindeki duyguyu analiz eder."
)

iface.launch()
