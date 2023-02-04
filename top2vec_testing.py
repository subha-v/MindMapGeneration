import re
#import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups
import numpy as np

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

#print(type(newsgroups.data))
print(newsgroups.data[0])

def generate_topics(article_text, num_topics=2, num_clusters=5):
    # Generate topics using Top2Vec
    model = Top2Vec(article_text, workers=4, embedding_model='universal-sentence-encoder', min_count=2)

    
    return model.topic_words

text_list = ["Benjamin Franklin's father, Josiah Franklin, was a tallow chandler, soaper, and candlemaker. Josiah Franklin was born at Ecton, Northamptonshire, England, on December 23, 1657, the son of Thomas Franklin, a blacksmith and farmer, and his wife, Jane White. Benjamin's father and all four of his grandparents were born in England. Josiah Franklin had a total of seventeen children with his two wives.", "Puppies are born after an average of 63 days of gestation, puppies emerge in an amnion that is bitten off and eaten by the mother dog. Puppies are born with a fully functional sense of smell. They are unable to open their eyes. During their first two weeks, a puppy's senses all develop rapidly."]
topic_words = generate_topics(text_list)
print(topic_words[0])
