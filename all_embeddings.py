import numpy as np
from scipy.spatial import KDTree
import nltk
from nltk.corpus import stopwords
import re
import os
from os.path import join, dirname

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

preprocessed_string = preprocessed_texts[0]

# Load the GloVe embeddings
glove_file = 'glove/glove.6B.50d.txt'
glove = {}
with open(glove_file, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove[word] = vector

# String containing all the words in the corpus

# Preprocess the corpus string into a list of words
corpus_words = preprocessed_string.split()

# Embed all the words using GloVe embeddings
corpus_embeddings = []
for word in corpus_words:
    if word in glove:
        embedding = glove[word]
        corpus_embeddings.append(embedding)

# Calculate the center of the corpus words
center = np.mean(corpus_embeddings, axis=0)

# Build a KDTree from all the words in the GloVe vocabulary
vocabulary = list(glove.keys())
vocabulary_embeddings = [glove[word] for word in vocabulary]
tree = KDTree(vocabulary_embeddings)

# Find the top 4 words in the total vocabulary that have the closest distances to the center of the corpus words
distances, indices = tree.query(center, k=5)  # k=5 because the closest point will always be the center itself
closest_words = [vocabulary[index] for index in indices]

# Print out those four words
print(closest_words[1:])  # exclude the first word (the center itself)