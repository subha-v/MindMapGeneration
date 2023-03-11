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

def main():
    # Download stopwords and punkt
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    data_root_folder = join(dirname(__file__), 'data')
    filenames = [
        join(data_root_folder, 'China', 'china.txt'),
    ]
    # TODO experiment by looping over different parameters
    for filename in filenames:
        get_main_topics(
            filename, n_clusters=2, embedding_size=50, norm_level=2
        )

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
        glove_embeddings = np.array([
            gloveword_to_embedding[word] for word in glovewords
        ])

        # plot_embeddings(corpus_embeddings, corpuswords)

        # TODO Experiment with different cluster sizes and plot how
        # good of a fit each has using matplotlib
        # The x axis will be the number of clusters you used
        # and the y axis will be the "goodness of fit".
        # Use the `.inertia_`

        # TODO make sure that 10 is a stable `n_init` value to use.
        # If you increase it and notice that the cluster sizes change,
        # then consider making this another experimental parameter to pass
        # to this function

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
        print(f'{centers.shape = }')

        # Find closest words in corpus to the centers
        closest_corpus_words = get_closest_words_to_centers(
            corpuswords, corpus_embeddings, centers, norm_level
        )
        closest_glove_words = get_closest_words_to_centers(
            glovewords, glove_embeddings, centers, norm_level
        )
        return closest_corpus_words, closest_glove_words, km.inertia_

def get_closest_words_to_centers(words, vectors, centers, norm_level):
    '''
    word: a list of `N` words
    vectors: array of `N` embeddings in `M` dimensions corresponding to words
    centers: `M` dimensional points
    norm_level: e.g. 1 if Manhattan distance, 2 if Euclidean, etc.
    '''
    tree = KDTree(vectors, p=norm_level)
    _, indices = tree.query(centers, k=10)

    # convert each element of `indices` into corresponding word
    closest_words = [
        [vectors[index] for index in cluster]
        for cluster in indices
    ]
    print('Words in corpus')
    for i, cluster in enumerate(closest_words):
        print(f'Cluster {i}: {cluster}')
    return closest_words

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
            word = values[0]
            # Skip adding stopwords or words that are a part of speech that we want to ignore
            # TODO skip words that are URLs or too long
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

if __name__ == '__main__':
    main()
