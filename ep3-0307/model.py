from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
<<<<<<< HEAD
from preprocess import preprocess_data 
=======
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC
import os
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_data, preprocess_data
import tensorflow as tf
>>>>>>> 27abdc5fc6a11441cffb8db78aa3a4349b2707f6

def build_model(vocab_size, max_len, output_dim=28):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len, mask_zero=True))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def build_model_aug(vocab_size, max_len, num_classes=28):
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        layers.SpatialDropout1D(0.5),
        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.LayerNormalization(),
        layers.Bidirectional(layers.GRU(64, return_sequences=False)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model

if __name__ == "__main__":
<<<<<<< HEAD
    print("Veriler işleniyor (preprocess.py)...")
    csv_path = "data/goemotions_merged.csv"
=======
    print("📦 Veriler yükleniyor ve temizleniyor...")
    csv_path = "data/augmented_dataset.csv"
>>>>>>> 27abdc5fc6a11441cffb8db78aa3a4349b2707f6
    max_len = 100

    # Yeni düzen: veri önce okunur, sonra işlenir
    texts, labels = load_data(csv_path)
    X_train, X_test, y_train, y_test, tokenizer, vocab_size = preprocess_data(
        texts=texts,
        labels=labels,
        max_words=10000,
        max_len=max_len
    )

<<<<<<< HEAD
    print("Model oluşturuluyor...")
    model = build_model(vocab_size=vocab_size, max_len=max_len)
=======
    print("🧠 Model oluşturuluyor...")
    model = build_model_aug(vocab_size=vocab_size, max_len=max_len)
>>>>>>> 27abdc5fc6a11441cffb8db78aa3a4349b2707f6

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

<<<<<<< HEAD
    print("Model kaydediliyor: goemotions_lstm_model.h5")
    model.save("goemotions_lstm_model.h5")

    # Skorları yazdır
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Eğitim tamamlandı.")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
=======
    os.makedirs("models", exist_ok=True)
    model_path = "models/goemotions_eda_model.h5"

    print(f"💾 Model kaydediliyor: {model_path}")
    model.save(model_path)

    train_acc = history.history.get('accuracy', [0])[-1]
    val_acc = history.history.get('val_accuracy', [0])[-1]
    val_auc = history.history.get('val_auc', [0])[-1]

    print(f"✅ EDA verisiyle eğitim tamamlandı. 🎉")
    print(f"📈 Train Accuracy: {train_acc:.4f}")
    print(f"📊 Validation Accuracy: {val_acc:.4f}")
    print(f"🧮 Validation AUC     : {val_auc:.4f}")
>>>>>>> 27abdc5fc6a11441cffb8db78aa3a4349b2707f6
