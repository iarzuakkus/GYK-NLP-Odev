# Proje Gelişim Raporu

## 1. Proje Amacı

Bu projede amaç, haber metinlerinden özgün özetler üretebilen bir doğal dil işleme (NLP) sistemi geliştirmektir. Transformer tabanlı bir mimari (T5-small) kullanılarak, CNN/DailyMail veri seti üzerinde model eğitilmiş ve test verileri üzerinde değerlendirilmiştir.

## 2. Kullanılan Veri Seti

* Veri Seti: CNN/DailyMail
* Alanlar: `article` (giriş metni) ve `summary` (referans özet)
* Eğitim verisi: `cnn_dailymail_sample.json` (10K örnek)
* Test verisi: `cnn_dailymail_test.json` (200 örnek)
* Doğrulama verisi: `cnn_dailymail_validation.json`

> Not: 1000 verilik ilk versiyon `t5-small-summary-v1` klasöründe; 10.000 verilik geliştirilmiş versiyon `t5-small-summary` klasöründe tutulmuş, farklar çıktı dosyalarında ("\_m2") belirtilmiştir.

## 3. Ön İşleme

* Metinler küçük harfe çevrilmiş ve temizlenmiştir.
* Tokenizer: `T5Tokenizer`
* Giriş metni: `summarize:` formatında hazırlanmıştır.
* Maksimum uzunluk: `max_length=128`
* Tokenize veriler: `tokenized_train_data.npz`, `tokenized_validation_data.npz`, `tokenized_data_1k.npz`

## 4. Model Eğitimi

* Model: `t5-small`
* Kod: `scripts/train.py`
* Parametreler: `outputs/hyperparameters_1k.json` ve `outputs/hyperparameters.json`
* Model dosyası: `models/t5-small-summary/` ve `t5-small-summary-v1/`

## 5. Tahmin Fonksiyonu

* Dosya: `scripts/predict.py`
* Amaç: Giriş metinlerinden özet üretimi
* Yalnızca yerel dosyalar kullanılmıştır (`local_files_only=True`)

## 6. Değerlendirme (ROUGE)

* Kullanılan metrikler: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
* Kod: `scripts/test.py`
* Test sonuçları: `outputs/test_results.json` ve `test_results_m2.json`

### Örnek Çıktı:

```json
{
  "index": 9,
  "article": "(CNN)For the first time in eight years, a TV legend returned to doing what he does best. Contestants told to \"come on down!\" on the April 1 edition of \"The Price Is Right\" encountered not host Drew Carey but another familiar face in charge of the proceedings. Instead, there was Bob Barker, who hosted the TV game show for 35 years before stepping down in 2007. Looking spry at 91, Barker handled the first price-guessing game of the show, the classic \"Lucky Seven,\" before turning hosting duties over to Carey, who finished up. Despite being away from the show for most of the past eight years, Barker didn't seem to miss a beat.",
  "reference_summary": "Bob Barker returned to host \"The Price Is Right\" on Wednesday .\nBarker, 91, had retired as host in 2007 .",
  "predicted_summary": "bob barker hosted the tv game show for 35 years before stepping down in 2007 he handled the first price-guessing game of the show, the classic \"lucky seven\""
}
```

### ROUGE Sonuçları:

**1K veri ile eğitilmiş model:**

* ROUGE-1: 0.3597
* ROUGE-2: 0.1183
* ROUGE-L: 0.2617
* ROUGE-Lsum: 0.3049

**10K veri ile eğitilmiş model:**

* ROUGE-1: 0.3613
* ROUGE-2: 0.1498
* ROUGE-L: 0.2666
* ROUGE-Lsum: 0.2987

> İlk modele göre 10K veri ile eğitilen model kelime çiftleri ve yapısal benzerlikte daha iyi performans göstermiştir.

## 7. Dosya Yapısı

```
ep5-1707/
├── data/                                # Ham veri dosyaları
│   ├── cnn_dailymail_sample.json        # 10K örnekten oluşan eğitim verisi
│   ├── cnn_dailymail_test.json          # 200 örnekten oluşan test verisi
│   └── cnn_dailymail_validation.json    # Doğrulama verisi
├── models/
│   ├── t5-small-summary/                # 10K veriyle eğitilmiş model dosyaları
│   └── t5-small-summary-v1/             # 1K veriyle eğitilmiş model versiyonu
├── outputs/                             # Eğitim ve test çıktıları
│   ├── hyperparameters_1k.json          # 1K eğitim parametreleri
│   ├── hyperparameters.json             # 10K eğitim parametreleri
│   ├── test_results.json                # 1K ROUGE test sonuçları
│   ├── test_results_m2.json             # 10K ROUGE test sonuçları
│   ├── tokenized_data_1k.npz            # Tokenize 1K veri
│   ├── tokenized_train_data.npz         # Tokenize eğitim verisi (10K)
│   ├── tokenized_validation_data.npz    # Tokenize doğrulama verisi
│   ├── train_log_1k.txt                 # 1K eğitim logları
│   └── train_log.txt                    # 10K eğitim logları
├── scripts/                             # Tüm işlem adımlarını içeren kodlar
│   ├── preprocessing.py                 # Metin ön işleme ve etiketleme
│   ├── predict.py                       # Eğitilen modelle özet üretme
│   ├── save_train_data.py               # datasets kütüphanesinden verileri çekme
│   ├── test_training_args.py            # Eğitim parametrelerini doğrulama aracı
│   ├── test.py                          # ROUGE skorları ile test işlemi
│   └── train.py                         # Model eğitimi
├── main.py                              # FastAPI tabanlı inference servisi
├── README.md                            # Proje hakkında genel açıklamalar
└── .gitignore                           # Git versiyon kontrolününe dahil edilmeyecek dosyalar
```

## 8. Sonuç

Model, sınırlı veriyle eğitilmesine rağmen çıktıları anlamlı ve tutarlıdır. ROUGE-1 ve ROUGE-Lsum skorları, kelime ölçeğinde yeterli kapsama sağladığını, ROUGE-2 skorları ise dil içi ardışık anlam bütünlüğünde hala geliştirilmeye açık noktalar olduğunu göstermektedir.

---

### Hazırlayanlar:

* İlayda Arzu AKKUŞ
* Ece Sude GÜNERHAN

GitHub: [GYK-NLP-Odev](https://github.com/iarzuakkus/GYK-NLP-Odev)
