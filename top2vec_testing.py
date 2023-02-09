from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups

def main():
    newsgroups = fetch_20newsgroups(
        subset='all', remove=('headers', 'footers', 'quotes')
    )
    model = Top2Vec(
        newsgroups.data, embedding_model='universal-sentence-encoder',
        min_count=50
    )

    # Get 3 topic groups for `newsgroups.data[0]`
    print(model.get_documents_topics([0], num_topics=3)[2])

if __name__ == '__main__':
    main()
