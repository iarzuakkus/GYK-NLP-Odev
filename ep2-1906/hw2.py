import sys
import os
import pandas as pd
from preprocessing import (
    preprocess_lowercase,
    preprocess_tokenize,
    preprocess_remove_stopwords,
)
from vectorization import vectorize_bow, vectorize_tfidf
from sentiment import analyze_sentiment_textblob
from topic_modeling import extract_topics_lda, extract_topics_nmf, name_topic


# 🔽 JSON'dan veri oku
def load_headlines(json_path: str, max_count: int = 10):
    df = pd.read_json(json_path, lines=True)
    return df["headline"].dropna().head(max_count).tolist()

if __name__ == "__main__":
    json_path = os.path.join(os.path.dirname(__file__), "news_category_dataset_v3.json")
    titles = load_headlines(json_path, max_count=30)

    cleaned_titles = []
    print("\n🔧 Preprocessing:\n")

    for title in titles:
        print(f"\n Orijinal: {title}")

        lower = preprocess_lowercase(title)
        print(f" Lowercase: {lower}")

        tokens = preprocess_tokenize(lower)
        print(f" Tokenized: {tokens}")

        no_stop = preprocess_remove_stopwords(lower)
        print(f" Stopwords Removed: {no_stop}")

        cleaned = " ".join(no_stop)
        print(f" Temiz Metin: {cleaned}")
        cleaned_titles.append(cleaned)

    print("\n Vektörleştirme:\n")

    bow_matrix, bow_features = vectorize_bow(cleaned_titles)
    print(" BoW Features:", bow_features)
    print("BoW Matrix:\n", bow_matrix)

    tfidf_matrix, tfidf_features = vectorize_tfidf(cleaned_titles)
    print("\n TF-IDF Features:", tfidf_features)
    print("TF-IDF Matrix:\n", tfidf_matrix)

    print("\n Duygu Analizi:\n")
    for title in titles:
        result = analyze_sentiment_textblob(title)
        print(f"{result['text']}\n→ Duygu: {result['sentiment']} (Polarity: {result['polarity']})\n")

    print("\n📚 Topic Modeling (LDA):\n")
    lda_topics = extract_topics_lda(cleaned_titles, n_topics=3)
    for name, words in lda_topics:
        label = name_topic(words)
        print(f"{name} ({label}): {', '.join(words)}")

    print("\n📚 Topic Modeling (NMF):\n")
    nmf_topics = extract_topics_nmf(cleaned_titles, n_topics=3)
    for name, words in nmf_topics:
        label = name_topic(words)
        print(f"{name} ({label}): {', '.join(words)}")

    

 