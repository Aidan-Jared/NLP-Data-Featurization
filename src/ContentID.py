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
    corpus = df.review_body.copy().values
    TextFormating(corpus)() #takes in the corpus and removes stop words, punct, and lems all the words
