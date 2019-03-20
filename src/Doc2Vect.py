import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import spacy
import random

np.random.seed(100)
class Doc2Vect(object):
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
    
    def __call__(self, save = False):
        self.documents = list(self.Tag_Document())
        model = self._make_model()
        if save:
            model.save('temp_Doc2Vec_model')
        doc_vec_train = [model.infer_vector(i) for i in self.X_train]
        doc_vec_test = [model.infer_vector(i) for i in self.X_test]
        doc_vec_train = np.vstack(doc_vec_train)
        doc_vec_test = np.vstack(doc_vec_test)
        return doc_vec_train, doc_vec_test
    def _make_model(self, embedding_size = 50):
        model = Doc2Vec(self.documents, vector_size=150, window=2, min_count=1)
        return model
    
    def Tag_Document(self):
        for idx, doc in enumerate(self.X_train):
            yield TaggedDocument(simple_preprocess(doc, min_len=1), [idx])