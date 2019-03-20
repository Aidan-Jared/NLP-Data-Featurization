import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from markdown import markdown
from bs4 import BeautifulSoup
from TextCleaning import TextFormating
import spacy

if __name__ == "__main__":
    table = pq.read_table('data/Amz_book_reveiws.c000.snappy.parquet')
    df = table.to_pandas()
    nlp = spacy.load('en_core_web_lg')

    mask = df['marketplace'] == 'US'
    df = df[mask].dropna()
    df_thing, df_debug = train_test_split(df,test_size=.001, random_state=42, stratify=df['star_rating'])

    for index, i in enumerate(df_debug['review_body']):
        html = markdown(i)
        Text = ' '.join(BeautifulSoup(html).findAll(text=True))
        corpus = TextFormating(Text, nlp)()
        df_debug['review_body'].iloc[index] = corpus
    df_debug.to_parquet('data/Amz_book_review_short_vector.parquet')