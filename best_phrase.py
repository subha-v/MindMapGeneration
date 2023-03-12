import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import re
import spacy

with open("data/China/china.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        #print(type(text))

def find_best_sentence(text, word):
    # Load the language model and process the text
    nlp = spacy.load('en_core_web_md')
    doc = nlp(text)

    # Find the most relevant sentence containing the word
    best_sentence = ""
    max_similarity = -1

    for sent in doc.sents:
        if word.lower() in sent.text.lower():
            similarity = sent.similarity(nlp(word))
            if similarity > max_similarity:
                best_sentence = sent.text
                max_similarity = similarity

    return best_sentence.strip()

print(find_best_sentence(text, "Emperor"))
