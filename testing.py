import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

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

vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b|\w+_+\w+')

X = vectorizer.fit_transform(preprocessed_texts)

# perform k-means clustering on the transformed data
num_clusters = 2  # choose the number of clusters to create
km = KMeans(n_clusters=num_clusters)
km.fit(X)

# assign each data point to a cluster
cluster_assignments = km.labels_

# determine the top words closest to the cluster center
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = list(vectorizer.vocabulary_.keys())
print(terms)

main_topics = []
for i in range(num_clusters):
    topic_words = []
    for j in order_centroids[i, :1]:
        topic_words.append(terms[j])
    main_topics.append(' | '.join(topic_words))
    
print('Main topics:')
for i, topic in enumerate(main_topics):
    print(f'Topic {i+1}: {topic}')

def load_glove_embeddings(embedding_file):
    embeddings = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def closest_subtopics(main_topics, corpus, embedding_file, num_top_words=5):
    # Load GloVe embeddings
    embeddings = load_glove_embeddings(embedding_file)

    # Tokenize the input corpus into a set of unique words
    corpus_words = set(re.findall(r'\b\w+\b', corpus.lower()))

    # Find the closest words to each main topic
    closest_words = []
    for main_topic in main_topics:
        # Get the embedding vector for the main topic
        if main_topic in embeddings:
            main_topic_embedding = embeddings[main_topic]
        else:
            print(f"Main topic '{main_topic}' not found in embeddings.")
            continue

        # Compute the cosine similarities between the main topic vector and all the other vectors in the corpus
        similarities = []
        for word in corpus_words:
            if word in embeddings and word != main_topic and word not in closest_words:
                word_embedding = embeddings[word]
                similarity = cosine_similarity(main_topic_embedding.reshape(1,-1), word_embedding.reshape(1,-1))[0,0]
                similarities.append((word, similarity))

        # Get the top num_top_words closest subtopics and their similarity scores
        top_words = [word for word, similarity in sorted(similarities, key=lambda x: x[1], reverse=True)[:num_top_words]]

        # Add the closest subtopics to the list
        closest_words.append(top_words)
        print(closest_words)

    return closest_words

closest_subtopics(["china", "tang"], " ".join(preprocessed_texts), "glove/glove.6B.50d.txt")
