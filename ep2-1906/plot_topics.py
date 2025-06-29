import matplotlib.pyplot as plt

def plot_lda_topics(model, feature_names, n_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        plt.figure(figsize=(10, 4))
        plt.bar(top_features, weights)
        plt.title(f"Topic {topic_idx + 1}")
        plt.ylabel("Kelime Ağırlığı")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
