import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import networkx as nx


### Getting the main topics

# download stopwords and punkt
nltk.download("stopwords")
nltk.download('punkt')

folder_path = "data/China/"
texts = []
for filename in os.listdir(folder_path):
    with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
        text = f.read()
        texts.append(text)


def preprocess_text(text):
    # remove non-alphabetic characters and lowercase the text
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    text = text.replace('_', ' ')
    # tokenize the text and remove stopwords
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords]

    return " ".join(filtered_tokens)


stopwords = set(stopwords.words("english"))
preprocessed_texts = []
for text in texts:
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords]
    preprocessed_texts = [preprocess_text(text) for text in texts]
    preprocessed_texts.append(" ".join(filtered_tokens))

print(type(preprocessed_texts))


# GETTING MAIN TOPICS

preprocessed_string = preprocessed_texts[0]


def top_4_words(text):
    # Split the input string into a list of words
    words = text.split()

    # Count the occurrences of each word in the list
    word_counts = Counter(words)

    # Get the top 4 most common words and return them as a list
    top_words = [word for word, count in word_counts.most_common(4)]
    return top_words


def top_n_words_tfidf(text, n=4):
    # Compute TF-IDF matrix for the input text
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])

    # Get the vocabulary from the vectorizer
    vocabulary = tfidf_vectorizer.vocabulary_

    # Get the inverse vocabulary, which maps indices to words
    inv_vocabulary = {v: k for k, v in vocabulary.items()}

    # Get the indices of the top n words based on their TF-IDF scores
    top_indices = np.argsort(tfidf_matrix.toarray()[0])[::-1][:n]

    # Get the corresponding words and return them as a list
    top_words = [inv_vocabulary[i] for i in top_indices]
    return top_words


main_topics = top_n_words_tfidf(preprocessed_string, n=4)
print("Main topics", main_topics)