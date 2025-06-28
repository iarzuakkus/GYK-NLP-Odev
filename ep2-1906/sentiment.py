from textblob import TextBlob

def analyze_sentiment_textblob(text: str) -> dict:
    """
    Temizlenmiş bir metin üzerinden duygu analizi yapar.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Pozitif" if polarity > 0 else "Negatif" if polarity < 0 else "Nötr"
    return {
        "text": text,
        "polarity": polarity,
        "sentiment": sentiment
    }
