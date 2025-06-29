import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Ece Sude Günerhan - İlayda Arzu Akkuş

'''
keys = ['punkt_tab','averaged_perceptron_tagger_eng','stopwords','wordnet']

for key in keys:
    nltk.download(key)
'''

def text_pipeline(process, corpus=''):
    if process == "tokenization":
        tokens = word_tokenize(corpus)
        return tokens

    elif process == "lowercasing":
        corpus = corpus.lower()
        return corpus

    elif process == "stopwords":
        stop_words = set(stopwords.words('english'))
        token = text_pipeline('tokenization', corpus)
        filtered_corpus = [word for word in token if word not in stop_words]
        return filtered_corpus

    elif process == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(corpus)
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized

    elif process == "tf-idf":
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(corpus)
        list = [vectorizer.get_feature_names_out(),X.toarray()]
        return list

    elif process == "generate":
        num = int(input("\nKaç cümle oluşturmak istiyorsunuz?: "))
        subjects = ["AI", "Artificial Intelligence", "Machine learning", "Deep learning", "NLP"]
        verbs = ["is transforming", "is used in", "is changing", "is improving", "is revolutionizing"]
        objects = [
            "healthcare", "the tech industry", "education", "customer service", 
            "finance", "transportation", "communication", "data analysis", "automation", "society"
        ]

        generated_corpus = []

        for _ in range(num):
            subject = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects)
            sentence = f"{subject} {verb} {obj}."
            generated_corpus.append(sentence)

        return generated_corpus


if __name__ == "__main__":

    corpus = [
    "Artificial Intelligence is the future.",
    "AI is changing the world.",
    "AI is a branch of computer science.",
    ]   

    processes = ['tokenization', 'lowercasing', 'stopwords', 'lemmatization']

    for process in processes:
        results = text_pipeline(process, corpus[0])
        print(f"\n--- {process.upper()} ---")
        print(results)

    # tf-idf işlemi
    results_tf_idf = text_pipeline('tf-idf', corpus)
    print("\n--- TF-IDF ---")
    print(results_tf_idf)

    # cümle üretimi
    results_generate = text_pipeline('generate')
    print("\n--- GENERATED SENTENCES ---")
    print(results_generate)
