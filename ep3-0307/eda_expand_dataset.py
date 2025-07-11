import random
from eda_utils import eda_augment
from preprocess import load_data  # CSV'den text ve label çeker
import os
import pandas as pd
import numpy as np

def expand_dataset_with_eda(texts, labels):
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        eda_versions = eda_augment(text)

        # 4 versiyondan rastgele 1 tanesini seç
        selected_aug = random.choice(eda_versions)

        augmented_texts.append(selected_aug)

        # Etiketin kopyasını al (NumPy array ise)
        if isinstance(label, np.ndarray):
            augmented_labels.append(label.copy())
        else:
            augmented_labels.append(label)

    # Final veri = orijinal + EDA’lı
    final_texts = texts + augmented_texts
    final_labels = list(labels) + augmented_labels  # orijinali de listeleştir

    return final_texts, final_labels

# ---------------- MAIN BLOK ---------------- #

if __name__ == "__main__":
    # 1. Veriyi yükle
    csv_path = "data/goemotions_merged.csv"  
    texts, labels = load_data(csv_path)

    # 2. EDA uygulayarak veri setini 2 katına çıkar
    texts_aug, labels_aug = expand_dataset_with_eda(texts, labels)

    print(len(texts_aug))
    print(len(labels_aug))

    # 3. DataFrame'e çevir
    df = pd.DataFrame(labels_aug)
    df.insert(0, 'text', texts_aug)  # ilk sütuna text ekle

    
    os.makedirs("data", exist_ok=True)

    
    df.to_csv("data/augmented_dataset.csv", index=False)
    print("✅ Veri 'data/augmented_dataset.csv' dosyasına kaydedildi.")
    print(f"Toplam örnek sayısı: {len(df)}")


    # 3. Kontrol çıktısı
    print(f"Orijinal örnek sayısı : {len(texts)}")
    print(f"EDA sonrası toplam   : {len(texts_aug)}")
    print(f"İlk örnek (EDA sonrası): {texts_aug[len(texts)]}")
