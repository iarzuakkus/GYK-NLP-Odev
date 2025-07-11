import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

# --- EDA Alt Fonksiyonları ---
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([w for w in words if wordnet.synsets(w)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if not synonyms:
            continue
        synonym_words = synonyms[0].lemma_names()
        if synonym_words:
            synonym = synonym_words[0].replace("_", " ")
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1 and counter < 10:
        random_word = random.choice(words)
        synonyms = wordnet.synsets(random_word)
        counter += 1
    if synonyms:
        synonym = synonyms[0].lemma_names()[0]
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, synonym)

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(words):
    if len(words) < 2:
        return words
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

def random_deletion(words, p):
    if len(words) == 1:
        return words
    return [w for w in words if random.uniform(0, 1) > p]

# --- Ana EDA Fonksiyonu ---
def eda_augment(text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
    words = text.split()
    num_words = len(words)

    augmented_texts = []

    a = max(1, int(alpha_sr * num_words))
    b = max(1, int(alpha_ri * num_words))
    c = max(1, int(alpha_rs * num_words))

    augmented_texts.append(" ".join(synonym_replacement(words, a)))
    augmented_texts.append(" ".join(random_insertion(words, b)))
    augmented_texts.append(" ".join(random_swap(words, c)))
    augmented_texts.append(" ".join(random_deletion(words, p_rd)))

    return augmented_texts  # Liste döner
