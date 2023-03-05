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

main_topics = top_4_words(preprocessed_string)
print("Main topics", main_topics)

# GETTING SUBTOPICS

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

closest_word_array = closest_subtopics(main_topics, " ".join(preprocessed_texts), "glove/glove.6B.200d.txt")

# FILTERING SUBTOPICS and MAINTOPICS

def filter_subtopics(main_topics, subtopics):
    # Filter out subtopics that are already in the list of main topics
    filtered_subtopics = []
    for subtopic_list in subtopics:
        filtered_subtopics.append([subtopic for subtopic in subtopic_list if subtopic not in main_topics])
    return filtered_subtopics

filtered_subtopics = filter_subtopics(main_topics, closest_word_array)

# GRAPH TOPICS

def create_graph(main_topics, subtopics):
    # Add main topics as blue nodes
    G = nx.Graph()
    for i, main_topic in enumerate(main_topics):
        G.add_node(main_topic, color='blue')

        # Add subtopics as pink nodes and connect them to their corresponding main topics
        for subtopic in subtopics[i]:
            G.add_node(subtopic, color='pink')
            G.add_edge(main_topic, subtopic)
    
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    plt.show()
    plt.savefig('graph.png')

create_graph(main_topics, filtered_subtopics)





