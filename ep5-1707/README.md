## Proje Gelişim Raporu

### 1. Proje Amacı

Bu projede amaç, haber metinlerinden özgün özetler üretebilen bir doğal dil işleme (NLP) sistemi geliştirmektir. Transformer tabanlı bir mimari (ozellikle T5-small) kullanılarak, CNN/DailyMail veri seti üzerinde model eğitilmiş ve eğitilen model daha sonra test verileri üzerinde değerlendirilmiştir.

### 2. Kullanılan Veri Seti

* **Veri Seti**: CNN/DailyMail
* **Alanlar**: `article` (giriş metni) ve `highlights` (referans özet)
* **Eğitim verisi**: İlk 1000 örnek (cnn\_dailymail\_sample.json)
* **Test verisi**: Sonraki 200 örnek (cnn\_dailymail\_test.json)

### 3. Ön İşleme

* Metinler küçük harfe çevrilmiş ve temizlenmiştir.
* **Tokenizer**: `T5Tokenizer`
* Giriş metni `summarize: ` şeklinde formatlanmıştır.
* Maksimum uzunluk: `max_length=128`

### 4. Model Eğitimi

* **Kullanılan Model**: `t5-small`
* Eğitim süreci `scripts/train.py` dosyası ile yürütülmüştür.
* Parametreler `outputs/hyperparameters.json` dosyasında tutulmuştur.
* Log dosyası: `train_log.txt`

### 5. Tahmin Fonksiyonu

* `scripts/predict.py` içinde tanımlanmıştır.
* Örnek metinlerden özet üretimi yapılmaktadır.
* Tokenizer ve model sadece yerel dosyalardan yüklenmektedir (`local_files_only=True`).

### 6. Değerlendirme (ROUGE)

* ROUGE metrikleri ile modelin başarımı test edilmiştir.
* Kullanılan metrikler: `ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `ROUGE-Lsum`
* Değerlendirme kodu: `scripts/test.py`

#### Örnek Çıktı

```json
{
  "index": 9,
  "article": "(CNN)For the first time in eight years...",
  "reference_summary": "Bob Barker returned to host ...",
  "predicted_summary": "bob barker hosted the tv game show ..."
}
```

#### ROUGE Sonuçları (10 örnek için):

* **ROUGE-1**: 0.3597
* **ROUGE-2**: 0.1183
* **ROUGE-L**: 0.2617
* **ROUGE-Lsum**: 0.3049

Model, 10.000 örnekle yeniden eğitildikten sonra ROUGE-1: 0.3613, ROUGE-2: 0.1498, ROUGE-L: 0.2666 ve ROUGE-Lsum: 0.2987 skorlarına ulaşarak bilgi kapsama açısından orta düzeyde başarı gösterse de, dil bütünlüğü ve yapı açısından hâlâ gelişime açıktır.

Bu skorlar, modelin kelime düzeyinde (ROUGE-1) makul bir başarım gösterdiğini, ancak daha karmaşık ardışık yapıları (ROUGE-2) yakalama konusunda sınırlı olduğunu göstermektedir.

### 7. Dosya Yapısı

```
ep5-1707/
|
├── data/                                # Örnek verilerin bulunduğu klasör
│   ├── cnn_dailymail_sample.json        # Eğitim için kullanılan 1000 örnek
│   ├── cnn_dailymail_test.json          # Testte kullanılan 200 örnek
│   └── cnn_dailymail_validation.json    # Opsiyonel validation verisi
│
├── models/
│   └── t5-small-summary/                # Model ve tokenizer dosyaları
│       ├── added_tokens.json            # Ekstra token'lar
│       ├── config.json                  # Model mimarisi
│       ├── generation_config.json       # Tahmin ayarları
│       ├── model.safetensors            # Model ağırlıkları
│       ├── special_tokens_map.json      # Token görevleri
│       ├── spiece.model                 # Tokenizer vocab
│       └── tokenizer_config.json        # Tokenizer davranış ayarları
│
├── outputs/
│   ├── hyperparameters.json             # Eğitim parametreleri
│   ├── test_results.json                # ROUGE skorları ve tahminler
│   ├── tokenized_data.npz               # Tokenize edilmiş numpy verisi
│   └── train_log.txt                    # Eğitim log dosyası
│
├── scripts/
│   ├── predict.py                       # Özetleme fonksiyonu
│   ├── preprocessing.py                 # Tokenizer ön işleme
│   ├── test.py                          # ROUGE skor testi
│   ├── train.py                         # Model eğitim scripti
│   └── test_training_args.py            # Parametre testi
│
├── main.py                              # FastAPI uygulama dosyası
├── README.md                            # Proje tanıtım dosyası
└── .gitignore                           # Git ignore ayarları
```

### 8. Sonuç ve Öneriler

Model, sınırlı veri ile eğitilmiş olmasına rağmen anlamlı ve tutarlı özetler üretebilmektedir. ROUGE-1 ve ROUGE-Lsum skorları bu başarıyı desteklemektedir. Daha büyük veri ile eğitim ve beam search gibi ünleme stratejileriyle modelin performansı daha da iyileştirilebilir.

---

**Hazırlayanlar:**
***İlayda Arzu Akkuş - Ece Sude Günerhan***
**GitHub**: [https://github.com/iarzuakkus/GYK-NLP-Odev/tree/main/ep5-1707](https://github.com/iarzuakkus/GYK-NLP-Odev/tree/main/ep5-1707)
