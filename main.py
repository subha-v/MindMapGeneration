import re
import os
from os.path import join, dirname

import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

def main():
    # TODO check if the download already exists and only run these
    # lines of code if so
    # Download stopwords and punkt
    nltk.download('stopwords')
    nltk.download('punkt')

    folder_path = join(dirname(__file__), 'data', 'Sleep')
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()

                sw = set(stopwords.words("english"))
                words = get_unique_words(text, sw)
                word_to_embedding = get_embeddings()
                words = [word for word in words if word in word_to_embedding]
                #breakpoint()
                print("Word to embedding values", len(word_to_embedding.values()))
                all_words = np.array(list(word_to_embedding.keys()))
                all_embeddings = np.array(list(word_to_embedding.values()))
                corpus_embeddings = np.array([
                    v for k, v in word_to_embedding.items() if k in words
                ])

                # Find closest words in corpus
                kmeans = KMeans(n_clusters=2, random_state=0).fit(corpus_embeddings)
                centers = kmeans.cluster_centers_
                tree = KDTree(corpus_embeddings)
                distances, indices = tree.query(centers, k=10)
                print("Words in text")
                for sublist in indices:
                    print([words[i] for i in sublist])

                # Find closest words among all embeddings
                kmeans = KMeans(n_clusters=2, random_state=0).fit(all_embeddings)
                centers = kmeans.cluster_centers_
                tree = KDTree(all_embeddings)
                distances, indices = tree.query(centers, k=10)
                print("All words indicies", indices)
                print("Words in all embeddings")
                for sublist in indices:
                    print([all_words[i] for i in sublist])

                #breakpoint()

def get_embeddings():
    word_to_embedding = {}
    with open(join(dirname(__file__), 'glove', 'glove.6B.50d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_to_embedding[word] = coefs
    return word_to_embedding

def get_unique_words(text, stopwords):
    # remove non-alphabetic characters and lowercase the text
    text = re.sub(r"[^a-z\s]", "", text.lower())
    # tokenize the text and remove stopwords
    tokens = nltk.word_tokenize(text)
    return sorted({token for token in tokens if token not in stopwords})

if __name__ == '__main__':
    main()
