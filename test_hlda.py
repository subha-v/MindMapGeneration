import string
import glob

from hlda.sampler import HierarchicalLDA
from nltk.corpus import stopwords

def main():
    stopset = stopwords.words('english') + list(string.punctuation) + ['will', 'also', 'said']

if __name__ == '__main__':
    main()
