import re
import os
from os.path import join, dirname
from collections import Counter
from nltk import pos_tag
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

# TODO Refactor code into functions so that you can easily run experiments

def main():
    # lines of code if so
    # Download stopwords and punkt
    # nltk.download('stopwords')
    # nltk.download('punkt')

    folder_path = join(dirname(__file__), 'data', 'China')
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()

                sw = set(stopwords.words("english"))

                # Get words from corpus as well as embeddings
                corpusword_to_freq = get_word_to_freq(text, sw)
                gloveword_to_embedding = get_embeddings()
                corpusword_to_freq = {
                    word: freq for word, freq in corpusword_to_freq.items()
                    if word in gloveword_to_embedding
                }

                corpuswords = sorted(corpusword_to_freq.keys())
                glovewords = sorted(gloveword_to_embedding.keys())

                print('Number of unique words in corpus: ', len(corpuswords))
                print('Number of word embeddings: ', len(gloveword_to_embedding))

                corpus_embeddings = np.array([
                    gloveword_to_embedding[word] for word in corpuswords
                ])
                glove_embeddings = np.array([
                    gloveword_to_embedding[word] for word in glovewords
                ])

                # TODO Experiment with different cluster sizes and plot how
                # good of a fit each has using matplotlib
                # The x axis will be the number of clusters you used
                # and the y axis will be the "goodness of fit".
                # Use the `.inertia_`
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(
                    corpus_embeddings,
                    sample_weight=np.array([
                        corpusword_to_freq[word] for word in corpuswords
                    ])
                )
                centers = kmeans.cluster_centers_
                print(f'{centers.shape = }')

                # TODO consider alternative metrics of similarity for the KDTree
                # (default is Euclidean)

                # Find closest words in corpus
                tree = KDTree(corpus_embeddings)
                distances, indices = tree.query(centers, k=10)
                print("Words in corpus")
                for j, sublist in enumerate(indices):
                    print(f'Cluster {j}: {[corpuswords[i] for i in sublist]}')

                # Find closest words among all embeddings
                tree = KDTree(glove_embeddings)
                distances, indices = tree.query(centers, k=10)
                print("Words in all embeddings")
                for j, sublist in enumerate(indices):
                    print(f'Cluster {j}: {[glovewords[i] for i in sublist]}')

                # TODO look into t-SNE which is a way of plotting word vectors
                # from original high dimenion (e.g. 50) to something that can
                # be viewd (e.g. 2 dimension)
                breakpoint()

def get_embeddings():
    sw = set(stopwords.words("english"))
    word_to_embedding = {}
    # TODO play with different embeddings sizes
    with open(join(dirname(__file__), 'glove', 'glove.6B.50d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
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
    tokens = nltk.word_tokenize(text)
    return Counter(token for token in tokens if token not in stopwords)

if __name__ == '__main__':
    main()
