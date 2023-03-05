import re
import os
from os.path import join, dirname
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def main():
    with open("data/China/china.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        sw = set(stopwords.words("english"))
        words = get_unique_words(text, sw)
        word_to_embedding = get_embeddings()
        words = [word for word in words if word in word_to_embedding]
        corpus_embeddings = np.array([
            v for k, v in word_to_embedding.items() if k in words
        ])

        # change words to a giant list and put it in an array
        words_string = " ".join(words)
        words_array = [words_string]

        # Print words array
        print("Words array", words_array)

        # Putting the countvectorize words into the word embedding
        main_topics = countvectorize_texts(words_array)
        
        print("Main topics", main_topics) 

        # Find closest words in corpus
        kmeans = KMeans(n_clusters=1, random_state=0).fit(corpus_embeddings)
        centers = kmeans.cluster_centers_
        print(centers.shape)
        tree = KDTree(corpus_embeddings)
        distances, indices = tree.query(centers, k=4)
        print("Words in text")
        for sublist in indices:
            print([words[i] for i in sublist])


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

def countvectorize_texts(texts):
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b|\w+_+\w+')

    X = vectorizer.fit_transform(texts)
    # perform k-means clustering on the transformed data
    num_clusters = 1  # choose the number of clusters to create
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)

    # assign each data point to a cluster
    cluster_assignments = km.labels_

    # determine the top words closest to the cluster center
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    main_topics = []
    for i in range(num_clusters):
        topic_words = []
        for j in order_centroids[i, :2]:
            topic_words.append(terms[j])
        main_topics.append(' | '.join(topic_words))
        
    print('Main topics:')
    for i, topic in enumerate(main_topics):
        print(f'Topic {i+1}: {topic}')

    return main_topics

if __name__ == '__main__':
    main()

