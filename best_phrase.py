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

#print(find_best_sentence(text, "Emperor"))

def generate_examples(text, word, num_examples=5):
    # Load the language model and process the text
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # Find sentences containing the word
    sentences = [sent for sent in doc.sents if word in sent.text.lower()]

    # Generate examples based on the context
    examples = []
    for sent in sentences:
        # Get the context of the word
        context = []
        for token in sent:
            if token.lower_ == word.lower():
                context.extend([t.text.lower() for t in token.lefts])
                context.append(token.text.lower())
                context.extend([t.text.lower() for t in token.rights])

        # Generate examples by replacing words in the context with similar words
        for i, context_word in enumerate(context):
            for token in doc:
                if token.text.lower() != word.lower() and token.text.lower() not in context and token.similarity(nlp(context_word)) > 0.2:
                    new_context = context[:i] + [token.text.lower()] + context[i+1:]
                    example = sent.text.lower()
                    for w in new_context:
                        example = example.replace(w, "____")
                    examples.append(example.capitalize() + ".")

                    if len(examples) >= num_examples:
                        return examples

    return examples

print(generate_examples(text, "Emperor"))