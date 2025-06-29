import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_topics import plot_lda_topics
from preprocessing import (
    preprocess_lowercase,
    preprocess_tokenize,
    preprocess_remove_stopwords,
    preprocess_lemmatize,
)
from vectorization import vectorize_bow, vectorize_tfidf
from sentiment import analyze_sentiment_textblob
from topic_modeling import extract_topics_lda, extract_topics_nmf, name_topic
from sentiment_tr import analyze_sentiment_turkish
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# JSON'dan veri oku
def load_headlines(json_path: str, max_count: int = 10):
    df = pd.read_json(json_path, lines=True)
    return df["headline"].dropna().head(max_count).tolist()

# LDA görselleştirme (matplotlib ile)
def plot_lda_topics(model, vectorizer, top_n=10):
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -top_n - 1 : -1]
        top_features = [words[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        plt.figure(figsize=(8, 4))
        plt.barh(top_features, weights)
        plt.gca().invert_yaxis()
        plt.title(f"Topic {topic_idx + 1}")
        plt.xlabel("Kelime Ağırlığı")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    json_path = os.path.join(os.path.dirname(__file__), "news_category_dataset_v3.json")
    titles = load_headlines(json_path, max_count=500)

    cleaned_titles = []
    print("\n Preprocessing:\n")

    for i, title in enumerate(titles[:5]):
        print(f"\n Orijinal: {title}")

        lower = preprocess_lowercase(title)
        print(f" Lowercase: {lower}")

        tokens = preprocess_tokenize(lower)
        print(f" Tokenized: {tokens}")

        no_stop = preprocess_remove_stopwords(lower)
        print(f" Stopwords Removed: {no_stop}")

        lemmatized = preprocess_lemmatize(" ".join(no_stop))
        print(f" Lemmatized: {lemmatized}")

        cleaned = " ".join(lemmatized)
        print(f" Temiz Metin: {cleaned}")
        cleaned_titles.append(cleaned)

    print("\n Vektörleştirme:\n")

    bow_matrix, bow_features = vectorize_bow(cleaned_titles)
    print(" BoW Features:", bow_features)
    print("BoW Matrix:\n", bow_matrix)

    tfidf_matrix, tfidf_features = vectorize_tfidf(cleaned_titles)
    print("\n TF-IDF Features:", tfidf_features)
    print("TF-IDF Matrix:\n", tfidf_matrix)

    print("\n BoW vs TF-IDF Karşılaştırması:\n")

    index = 0
    print(" Orijinal Başlık:", titles[index])
    print(" Temizlenmiş:", cleaned_titles[index])

    print("\n BoW Vektörü:")
    bow_vector = bow_matrix[index]
    for word, val in zip(bow_features, bow_vector):
        if val > 0:
            print(f" {word}: {val}")

    print("\n TF-IDF Vektörü:")
    tfidf_vector = tfidf_matrix[index]
    for word, val in zip(tfidf_features, tfidf_vector):
        if val > 0:
            print(f" {word}: {val:.3f}")

    print("\n Yöntem Karşılaştırması:\n")

    print("BoW (Bag of Words):")
    print("- + Basit ve hızlıdır.")
    print("- – Her kelimenin sadece geçme sayısını sayar, önem derecesini bilmez.")
    print("- – Yaygın kelimelerle nadir kelimeleri ayıramaz.")

    print("\nTF-IDF (Term Frequency - Inverse Document Frequency):")
    print("- + Nadir ama ayırt edici kelimelere daha fazla ağırlık verir.")
    print("- + Bilgilendirici kelimelere öncelik tanır.")
    print("- – Hesaplama maliyeti biraz daha yüksektir.")

    print("\n Duygu Analizi:\n")
    for i, title in enumerate(titles[:10]):
        result = analyze_sentiment_textblob(title)
        print(f"{result['text']}\n→ Duygu: {result['sentiment']} (Polarity: {result['polarity']})\n")

    print("\n Topic Modeling (NMF):\n")
    nmf_topics = extract_topics_nmf(cleaned_titles, n_topics=3)
    for name, words in nmf_topics:
        label = name_topic(words)
        print(f"{name} ({label}): {', '.join(words)}")

    print("\n Topic Modeling (LDA):\n")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    lda_dtm = vectorizer.fit_transform(cleaned_titles)
    lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
    lda_model.fit(lda_dtm)

    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [words[i] for i in top_indices]
        label = name_topic(top_words)
        print(f"Topic {i+1} ({label}): {', '.join(top_words)}")

    # Görselleştir
    plot_lda_topics(lda_model, vectorizer, top_n=10)

    print("\n Türkçe Başlık Duygu Analizi:\n")
    turkish_titles = [
        "Enflasyon rakamları beklentileri aştı",
        "Milli takım galibiyetle turladı",
        "Yeni vergi düzenlemesi vatandaşları üzdü",
        "Hava sıcaklığı mevsim normallerinin üzerine çıkıyor",
        "Türkiye ekonomisi toparlanma sinyalleri veriyor"
    ]

    for title in turkish_titles:
        result = analyze_sentiment_turkish(title)
        print(f"{result['text']} → {result['sentiment']} ({result['stars']} yıldız)")

'''Topic Modeling Çıktılarını Yorumlama ve İyileştirme Önerileri

Topic modeling çıktılarında bazı konu kümeleri anlamlı görünse de, genel olarak "dy", "23", "year", "air" gibi bağlamdan kopuk kelimelerle karışık konular elde edilmiştir. 
Bunun temel nedeni, küçük veri setiyle (30 başlık) çalışmak ve bazı preprocessing işlemlerinin bağlamı daraltmasıdır. Bu çıktılar, LDA ve NMF algoritmalarının yeterli bağlamda çalışamadığını göstermektedir. 
Daha anlamlı topic kümeleri için veri miktarı artırılmalı, n-gram desteği eklenmeli ve TF-IDF filtreleme parametreleri yeniden ayarlanmalıdır.'''
