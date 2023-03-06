import numpy as np
from scipy.spatial import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import nltk
from nltk.corpus import stopwords
import re
nltk.download('punkt')

glove_path = 'glove/glove.6B.50d.txt'

# Define the function to load the GloVe embeddings
def load_glove_embeddings(embedding_file):
    embeddings = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load the GloVe embeddings
embeddings = load_glove_embeddings(glove_path)

# Define the function to calculate the TF-IDF weighted embedding of a document
def tfidf_weighted_embedding(document, embeddings):
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    tfidf.fit_transform([document])
    word2weight = dict(zip(tfidf.vocabulary_.keys(), tfidf.idf_))
    weighted_embedding = np.zeros((embeddings.vector_size,), dtype=np.float32)
    for word in document:
        if word in embeddings.vocab and word in word2weight:
            weighted_embedding += embeddings[word] * word2weight[word]
    return weighted_embedding



# Define the function to find the closest words in the corpus to the center via GloVe embeddings and TF-IDF weightage
def closest_words_to_center(text, embeddings, n=4):
    print("Started closest words to center")
    #text_string = ' '.join(text)
    weighted_embeddings = []
    for doc in text:
        weighted_embedding = tfidf_weighted_embedding(doc, embeddings)
        weighted_embeddings.append(weighted_embedding)
    weighted_embeddings = np.array(weighted_embeddings)
    print("Ended weighted embeddings")
    center_embedding = np.mean(weighted_embeddings, axis=0)
    kdtree = KDTree(weighted_embeddings)
    dist, indices = kdtree.query([center_embedding], k=n)
    closest_words = []
    for index in indices[0]:
        closest_words.append(text[index])
    return closest_words

def cleaning(text):
        # remove non-alphabetic characters and lowercase the text
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        text = text.replace('_', ' ')
        # tokenize the text and remove stopwords
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        return " ".join(filtered_tokens)

# Define the text to analyze
folder_path = "data/China/"
texts = []
for filename in os.listdir(folder_path):
    with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
        text = f.read()
        texts.append(text)

preprocessed_texts = [cleaning(text) for text in texts]
preprocessed_string = preprocessed_texts[0]
preprocessed_tokens = [nltk.word_tokenize(text) for text in preprocessed_texts]

# Find the closest words in the corpus to the center via GloVe embeddings and TF-IDF weightage
closest_words = closest_words_to_center(preprocessed_tokens, embeddings)
print(closest_words)