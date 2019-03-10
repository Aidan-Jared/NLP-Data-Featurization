import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import spacy
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from wordcloud import WordCloud

if __name__ == "__main__":
    #importing files and making the main corpus to work with
    table = pq.read_table('data/Amz_book_review_short.parquet')
    df = table.to_pandas()
    corpus = df.review_body.values

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(corpus)
    # stopwords = text.ENGLISH_STOP_WORDS
    # vect = CountVectorizer(max_features=500, max_df=.85, min_df=2, stop_words=stopwords)
    # vect.fit(corpus)