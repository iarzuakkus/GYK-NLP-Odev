from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Tuple
import numpy as np
import sys
import os

# ep1-1706 klasörünü modül arama yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
hw1_path = os.path.abspath(os.path.join(current_dir, "..", "ep1-1706"))
sys.path.append(hw1_path)
from hw1 import text_pipeline  # type: ignore

def vectorize_bow(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(texts)
    return X.toarray(), vectorizer.get_feature_names_out().tolist()

def vectorize_tfidf(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    result = text_pipeline("tf-idf", texts)
    features, matrix = result[0], result[1]
    return matrix, features