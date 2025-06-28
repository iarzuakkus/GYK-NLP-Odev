from sklearn.decomposition import LatentDirichletAllocation, NMF
from vectorization import vectorize_bow, vectorize_tfidf

def extract_topics_lda(texts, n_topics=3, n_words=10):
    """
    BoW vektörlerinden LDA ile topic çıkarımı yapar.
    """
    bow_matrix, features = vectorize_bow(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(bow_matrix)

    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [features[i] for i in topic.argsort()[-n_words:]]
        topics.append((f"Topic {idx+1}", top_words[::-1]))

    return topics

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
