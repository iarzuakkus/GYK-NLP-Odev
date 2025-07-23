# Proje Adı

**Kısa Açılama:**  
Bu proje, metin özetleme (örneğin haber makaleleri) için bir NLP modeli eğitmek ve kullanmak amacıyla oluşturulmuştur. T5 (`t5-small-summary`) modeli kullanılarak eğitim ve öngörü işlemleri yapılır.

---

## 📂 Dosya Yapısı
proje/
├── data/
│ └── cnn_dailymail_sample.json # Örnek veri seti (CNN/DailyMail)
├── models/
│ └── t5-small-summary/ # Fine-tuning edilmiş T5 modeli
│ ├── added_tokens.json
│ ├── config.json
│ ├── model.safetensors
│ └── ... (diğer model dosyaları)
├── outputs/ # Eğitim çıktıları
│ ├── hyperparameters.json
│ ├── tokenized_data.npz
│ └── train_log.txt
├── scripts/
│ ├── predict.py # Metin özetleme için betik
│ └── preprocessing.py # Veri ön işleme betiği
├── test_training_args.py # Eğitim argümanlarını test etme
├── train.py # Model eğitim betiği
├── main.py # Ana uygulama betiği
├── README.md # Bu dosya
└── .gitignore


---

##  Kurulum

1. **Gereksinimler:**  
   Python 3.8+ ve gerekli kütüphaneler:
   ```bash
   pip install transformers torch numpy datasets
Model ve Veri:

models/t5-small-summary klasörüne fine-tuning edilmiş model dosyalarını yerleştirin.

Veri setini data/cnn_dailymail_sample.json içine ekleyin.

 Kullanım
🔧 Eğitim
bash
python train.py  # Hyperparameters.json'dan ayarları okuyarak eğitimi başlatır
 Özetleme (Prediction)
bash
python scripts/predict.py --input_text "Özetlenecek metin buraya..."
 Ön İşleme
bash
python scripts/preprocessing.py  # Veriyi tokenize eder ve .npz formatında kaydeder
 Özellikler
Model: t5-small (Hafif ve hızlı özetleme için optimize edilmiş).

Veri Seti: CNN/DailyMail örnekleri.

Çıktılar: Eğitim logları, tokenize edilmiş veriler ve hiperparametreler.

 Lisans
Bu proje MIT Lisansı altında dağıtılmaktadır.

text

---

###  Notlar:
- `main.py` ve `predict.py` gibi dosyaların içeriğine bağlı olarak **Kullanım** kısmını genişletebilirsiniz.  
- Modelin performansı veya eğitim detayları için `outputs/train_log.txt` dosyasını inceleyin.  
- Projeye özel kurulum adımları (örn. GPU kullanımı) eklemek isterseniz, **Kurulum** bölümünü güncelleyin.