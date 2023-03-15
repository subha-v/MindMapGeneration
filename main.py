import re
from os.path import join, dirname
from collections import Counter
from nltk import pos_tag
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import csv
import itertools

def main():
    # Download stopwords and punkt if necessary
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')

    data_root_folder = join(dirname(__file__), 'data')
    filename = join(data_root_folder, 'China', 'china.txt')

    norm_levels = [1,2,3,6,10,100,1000]
    # with open('output.csv', mode='w') as csv_file:
    #     fieldnames = ['norm_level', 'averaged_distance', 'km_inertia']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    for norm_level in norm_levels:
        results = get_main_topics(filename, n_clusters=1, embedding_size=50, norm_level=norm_level)
        # writer.writerow({'norm_level': norm_level, 'averaged_distance': results[0]})
        print(f"Processed file {filename}: norm_level={norm_level}")
        print({'norm_level': norm_level, 'averaged_distance': results})

def get_main_topics(filename, n_clusters, embedding_size, norm_level):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

        sw = set(stopwords.words("english"))

        # Get words from corpus as well as embeddings
        corpusword_to_freq = get_word_to_freq(text, sw)
        gloveword_to_embedding = get_embeddings(sw, embedding_size)

        # Filter out words in corpus that have no GloVe embedding
        corpusword_to_freq = {
            word: freq for word, freq in corpusword_to_freq.items()
            if word in gloveword_to_embedding
        }

        corpuswords = sorted(corpusword_to_freq.keys())
        glovewords = sorted(gloveword_to_embedding.keys())

        print('Number of unique words in corpus: ', len(corpuswords))
        print('Number of word embeddings: ', len(gloveword_to_embedding))

        # Create matrices with each row corresponding to a word, with words
        # sorted alphabetically
        corpus_embeddings = np.array([
            gloveword_to_embedding[word] for word in corpuswords
        ])

        # Set up kmeans by weighting words by the number of times it appeared
        # in the corpus
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        km.fit(
            corpus_embeddings,
            sample_weight=np.array([
                corpusword_to_freq[word] for word in corpuswords
            ])
        )
        centers = km.cluster_centers_

        # Calculate averaged distances
        distances = []
        for center in centers:
            closest_words = get_closest_words_to_centers(
                corpuswords, corpus_embeddings, center.reshape(1,-1), norm_level, True
            )
            closest_words = list(itertools.chain.from_iterable(closest_words))
            closest_word_embeddings = np.array([
                gloveword_to_embedding[word] for word in closest_words
            ])
            
            
            distances.append(np.mean(np.linalg.norm(closest_word_embeddings-center, axis=1)))
        averaged_distance = np.mean(distances)
        print("Kmeans inertia", km.inertia_)
        km = None

        return averaged_distance

def get_closest_subtopics(topic_list, filename, embedding_size):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

        sw = set(stopwords.words("english"))

        # Get words from corpus as well as embeddings
        corpusword_to_freq = get_word_to_freq(text, sw)
        gloveword_to_embedding = get_embeddings(sw, embedding_size)

        # Filter out words in corpus that have no GloVe embedding
        corpusword_to_freq = {
            word: freq for word, freq in corpusword_to_freq.items()
            if word in gloveword_to_embedding
        }

        corpuswords = sorted(corpusword_to_freq.keys())
        glovewords = sorted(gloveword_to_embedding.keys())

        print('Number of unique words in corpus: ', len(corpuswords))
        print('Number of word embeddings: ', len(gloveword_to_embedding))

        # Create matrices with each row corresponding to a word, with words
        # sorted alphabetically
        corpus_embeddings = np.array([
            gloveword_to_embedding[word] for word in corpuswords
        ])
        glove_embeddings = np.array([
            gloveword_to_embedding[word] for word in glovewords
        ])
        similar_words = []
        for sublist in topic_list:
            similar_sublist = []
            for word in sublist:
                if word in corpuswords:
                    word_embedding = corpus_embeddings[corpuswords.index(word)].reshape(1, -1)
                    similarities = cosine_similarity(word_embedding, corpus_embeddings)[0]
                    closest_word_indices = np.argsort(similarities)[::-1][1:6] # exclude the word itself
                    closest_words = [corpuswords[i] for i in closest_word_indices]
                    similar_sublist.append(closest_words)
                else:
                    word_embedding = glove_embeddings[glovewords.index(word)].reshape(1, -1)
                    similarities = cosine_similarity(word_embedding, glove_embeddings)[0]
                    closest_word_indices = np.argsort(similarities)[::-1][1:6] # exclude the word itself
                    closest_words = [glovewords[i] for i in closest_word_indices]
                    similar_sublist.append(closest_words)
            similar_words.append(similar_sublist)
        return similar_words

def filter_subtopics(main_topics, subtopics):
    # Filter out subtopics that are already in the list of main topics
    filtered_subtopics = []
    for subtopic_list in subtopics:
        filtered_subtopics.append([subtopic for subtopic in subtopic_list if subtopic not in main_topics])
    return filtered_subtopics

def get_closest_words_to_centers(words, vectors, centers, norm_level, is_corpus):
    '''
    word: a list of `N` words
    vectors: array of `N` embeddings in `M` dimensions corresponding to words
    centers: `M` dimensional points
    norm_level: e.g. 1 if Manhattan distance, 2 if Euclidean, etc.
    is_corpus: whether the text is from the corpus or from GloVe
    '''
    tree = KDTree(vectors)
    distances, indices = tree.query(centers, k=4, p=norm_level)
    print("Distances", distances)

    # convert each element of `indices` into corresponding word
    closest_words = [
        [words[index] for index in cluster]
        for cluster in indices
    ]
    if(is_corpus==True):
        print('corpus: ', closest_words)
    else:
        print('glove: ', closest_words)
    
    return closest_words

def clean_word(word):
    # remove any URLs in the word
    word = re.sub(r'http\S+', '', word)
    # remove any non-word characters from the beginning and end of the word
    word = re.sub(r'^\W+|\W+$', '', word)
    # remove any random strings of characters that are not words
    return word

def get_embeddings(sw, embedding_size=50):
    valid_embedding_sizes = {50, 100, 200, 300}
    if embedding_size not in valid_embedding_sizes:
        raise ValueError(
            f'The embeddings size {embedding_size} is not among the available '
            f'sizes of {valid_embedding_sizes}'
        )

    word_to_embedding = {}
    with open(join(dirname(__file__), 'glove', f'glove.6B.{embedding_size}d.txt')) as f:
        for line in f:
            values = line.split()
            word = clean_word(values[0])
            if word is None or any(char.isdigit() for char in word):
                continue
            # Skip adding stopwords or words that are a part of speech that we want to ignore
            if word in sw or pos_tag([word])[0][1] in ('DT', 'PRP', 'PRP$', 'IN', 'CC', 'TO'):
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            word_to_embedding[word] = coefs
    return word_to_embedding

def get_word_to_freq(text, stopwords):
    # remove non-alphabetic characters and lowercase the text
    text = re.sub(r"[^a-z\s]", "", text.lower())
    # tokenize the text and remove stopwords
    # TODO filter the corpus words with POS tagging
    tokens = nltk.word_tokenize(text)
    return Counter(token for token in tokens if token not in stopwords)

def plot_embeddings(embeddings, words):
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(16, 16))
    for i, word in enumerate(words):
        x, y = embeddings_2d[i, :]
        ax.scatter(x, y, color='blue')
        ax.annotate(word, xy=(x, y), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom', fontsize=14)
    plt.show()

def compute_silhouette_score(embeddings, cluster_centers):
    """
    Computes the Silhouette coefficient for k-NN clustering given a vector of embeddings,
    a list of cluster centers, and the number of clusters (k).
    """
    labels = []  # list of cluster labels for each data point
    for emb in embeddings:
        # compute the distance between the embedding and each cluster center
        distances = [np.linalg.norm(emb - c) for c in cluster_centers]
        # assign the embedding to the cluster with the nearest center
        label = np.argmin(distances)
        labels.append(label)
    # compute the Silhouette coefficient for the clustering
    return silhouette_score(embeddings, labels)

if __name__ == '__main__':
    main()
