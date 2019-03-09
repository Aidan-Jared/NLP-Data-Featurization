import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from markdown import markdown
from bs4 import BeautifulSoup

if __name__ == "__main__":
    run_full = bool(input('run full: '))
    #reading in parquet file into pandas dataframe
    table = pq.read_table('data/Amz_book_reveiws.c000.snappy.parquet')
    df = table.to_pandas()

    #only select the US marketplace
    mask = df['marketplace'] == 'US'
    df = df[mask]

    # Cleaning out markdown formating
    if run_full != True:
        df_debug, df_test = train_test_split(df, test_size = .2, random_state=42)
        for index, i in enumerate(df_debug['review_body']):
            html = markdown(i)
            Text = ' '.join(BeautifulSoup(html).findAll(text=True))
            df_debug['review_body'].iloc[index] = Text
        df_debug.to_csv('data/Amz_book_review_short.csv')
    else:
        for index, i in enumerate(df['review_body']):
            html = markdown(i)
            Text = ' '.join(BeautifulSoup(html).findAll(text=True))
            df['review_body'].iloc[index] = Text
        df.to_csv('data/Amz_book_review_full.csv')