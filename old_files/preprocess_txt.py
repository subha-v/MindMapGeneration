import os
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')


def preprocess_text(path):
    folder_path = path
    texts = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            texts.append(text)

    stopwords2 = set(stopwords.words("english"))
    preprocessed_texts = []
    print(texts)
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stopwords2]
        preprocessed_texts = [clean_text(text) for text in texts]
        preprocessed_texts.append(" ".join(filtered_tokens))

    preprocessed_string = preprocessed_texts[0]
    return preprocessed_string

def clean_text(text):
        # remove non-alphabetic characters and lowercase the text
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        text = text.replace('_', ' ')
        # tokenize the text and remove stopwords
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stopwords]
        return " ".join(filtered_tokens)