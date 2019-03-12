import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class TextFormating(object):
    '''
    
    '''
    def __init__(self, X):
        self.X = X
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(self._textClean, last = True, name='TextClean')
        self.tokenizer = spacy.tokenizer.Tokenizer(self.nlp.vocab)

    def __call__(self):
        for index, i in enumerate(self.X):
            self.X[index] = self.nlp(i)
             
    def _textClean(self, Doc):
        lem = []
        for i in Doc:
            if i.is_punct != True:
                if i.is_stop != True and i.is_space != True:
                    lem.append(i.lemma_)
        return self.tokenizer(' '.join(lem))