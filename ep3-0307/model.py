from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import preprocess_data 

def build_model(vocab_size, max_len, output_dim=28):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len, mask_zero=True))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    print("Veriler işleniyor (preprocess.py)...")
    csv_path = "data/goemotions_merged.csv"
    max_len = 100

    X_train, X_test, y_train, y_test, tokenizer, vocab_size = preprocess_data(
        csv_path=csv_path,
        max_words=10000,
        max_len=max_len
    )

    print("Model oluşturuluyor...")
    model = build_model(vocab_size=vocab_size, max_len=max_len)

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

    print("Model kaydediliyor: goemotions_lstm_model.h5")
    model.save("goemotions_lstm_model.h5")

    # Skorları yazdır
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Eğitim tamamlandı.")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
