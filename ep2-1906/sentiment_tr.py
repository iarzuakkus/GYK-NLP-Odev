from transformers import pipeline

# Yalnızca PyTorch kullan
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    framework="pt"  # ← Bu satır kritik!
)

def analyze_sentiment_turkish(text: str) -> dict:
    result = sentiment_pipeline(text)[0]
    label = result["label"]  # örnek: '5 stars'
    stars = int(label[0])

    if stars >= 4:
        sentiment = "Pozitif"
    elif stars == 3:
        sentiment = "Nötr"
    else:
        sentiment = "Negatif"

    return {
        "text": text,
        "stars": stars,
        "sentiment": sentiment
    }
