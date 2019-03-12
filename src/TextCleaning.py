import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def textClean(Doc):
    lem =[]
    stopWords = spacy.lang.en.stop_words.STOP_WORDS
    for i in Doc:
        if i.is_punct != True:
            if i.lemma_ not in stopWords and i.is_space != True:
                lem.append(i.lemma_)
    return " ".join(lem)


class TextFormating(object):
    def __init__(self, X, FullClean = True):
        self.X = X
        self.FullClean = FullClean
    def __call__(self):
        nlp = spacy.load('en_core_web_sm')
        if self.FullClean == True:
            nlp.add_pipe(textClean, last = True, name='TextClean')
        elif self.FullClean == False:
            if nlp.pipeline[-1][0] == "TextClean":
                nlp.remove_pipe('TextClean')
        for index, i in enumerate(self.X):
            self.X[index] = nlp(i)
        return self.X