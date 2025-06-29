from sklearn.decomposition import LatentDirichletAllocation, NMF
from vectorization import vectorize_bow, vectorize_tfidf

from sklearn.decomposition import LatentDirichletAllocation
from vectorization import vectorize_bow

import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def extract_topics_lda(texts, n_topics=5, n_words=7):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    topics = []
    for topic in lda.components_:
        top_indices = topic.argsort()[-n_words:][::-1]
        words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        topics.append(words)

    return lda, topics, dtm, vectorizer 


def extract_topics_nmf(texts, n_topics=3, n_words=10):
    """
    TF-IDF vektörlerinden NMF ile topic çıkarımı yapar.
    """
    tfidf_matrix, features = vectorize_tfidf(texts)

    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf_matrix)

    topics = []
    for idx, topic in enumerate(nmf.components_):
        top_words = [features[i] for i in topic.argsort()[-n_words:]]
        topics.append((f"Topic {idx+1}", top_words[::-1]))

    return topics

def name_topic(top_words: list) -> str:
    """
    Temsilci kelimelere göre genel haber kategorisi atar.
    """
    topic_keywords = {
        "Teknoloji": ["ai", "tech", "robot", "software", "hardware", "digital", "gadget"],
        "Bilim": ["data", "research", "experiment", "science", "study"],
        "Ekonomi": ["finance", "market", "stock", "economy", "business", "inflation", "bank", "money"],
        "Eğitim": ["school", "student", "university", "teacher", "exam", "college"],
        "Sağlık": ["health", "doctor", "covid", "vaccine", "hospital", "virus"],
        "Politika": ["election", "government", "senate", "law", "president", "politician"],
        "Toplum / Güvenlik": ["police", "crime", "violence", "court", "attack", "shooting"],
        "Ulaşım": ["flight", "airlines", "train", "travel", "bus", "airport"],
        "Dünya": ["refugee", "border", "war", "global", "nation", "international"],
        "Spor": ["sports", "game", "match", "player", "team", "league"],
        "Eğlence": ["movie", "film", "celebrity", "music", "tv", "series"],
        "Sosyal Medya": ["tweet", "twitter", "video", "viral", "platform", "hashtag"]
    }

    for word in top_words:
        word = word.lower()
        for category, keywords in topic_keywords.items():
            if word in keywords:
                return category

    return "Diğer / Belirsiz"

def visualize_lda_model(model, dtm, vectorizer, output_html="lda_viz.html"):
    """
    model: sklearn LDA modeli
    dtm: belge-terim matrisi (TF-IDF veya Count)
    vectorizer: kullanılan vectorizer (CountVectorizer veya TfidfVectorizer)
    output_html: HTML olarak kayıt edilecek dosya ismi
    """

    panel = pyLDAvis.sklearn.prepare(model, dtm, vectorizer)
    pyLDAvis.save_html(panel, output_html)
    print(f"\n LDA görselleştirme başarıyla '{output_html}' dosyasına kaydedildi.")


