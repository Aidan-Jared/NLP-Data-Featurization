import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from wordcloud import WordCloud

if __name__ == "__main__":
    #importing files and making the main corpus to work with
    df = pd.read_csv('data/Amz_book_review_short.csv')
    df.dropna(inplace=True)
    corpus = df['reveiw_body'].values

    stopwords = text.ENGLISH_STOP_WORDS
    vect = CountVectorizer(max_features=500, max_df=.85, min_df=2, stop_words=stopwords)