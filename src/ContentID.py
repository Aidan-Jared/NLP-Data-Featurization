import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from wordcloud import WordCloud
from TextCleaning import TextFormating

if __name__ == "__main__":
    #importing files and making the main corpus to work with
    table = pq.read_table('data/Amz_book_review_short.parquet')
    df = table.to_pandas()
    corpus = df.review_body.values

    corpus_FC = TextFormating(corpus, FullClean=True).__call__()
    # for index, i in enumerate(corpus):
    #     corpus[index] = nlp(i)
    # stopwords = text.ENGLISH_STOP_WORDS
    # vect = CountVectorizer(max_features=500, max_df=.85, min_df=2, stop_words=stopwords)
    # vect.fit(corpus)