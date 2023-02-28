import os
from os.path import join, dirname
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re

# download stopwords and punkt
nltk.download('stopwords')
nltk.download('punkt')

folder_path = join(dirname(__file__), 'data', 'China')
texts = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)

def preprocess_text(text, stopwords):
    # remove non-alphabetic characters and lowercase the text
    text = re.sub(r"[^a-z\s]", "", text.lower())
    # tokenize the text and remove stopwords
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return " ".join(filtered_tokens)

stopwords = set(stopwords.words("english"))
preprocessed_texts = [preprocess_text(text, stopwords) for text in texts]

