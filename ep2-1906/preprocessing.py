import sys
import os
from typing import List, Union

# ep1-1706 klasörünü modül arama yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
hw1_path = os.path.abspath(os.path.join(current_dir, "..", "ep1-1706"))
sys.path.append(hw1_path)
from hw1 import text_pipeline  # type: ignore

def preprocess_lowercase(text: str) -> str:
    return text_pipeline("lowercasing", text)

def preprocess_tokenize(text: str) -> List[str]:
    return text_pipeline("tokenization", text)

def preprocess_remove_stopwords(text: str) -> List[str]:
    return text_pipeline("stopwords", text)

def preprocess_lemmatize(text: str) -> Union[str, List[str]]:
    return text_pipeline("lemmatization", text)


