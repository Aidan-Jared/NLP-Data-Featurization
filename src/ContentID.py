import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from ModelMaker import ModelMaker
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from TextCleaning import TextFormating

if __name__ == "__main__":
    #importing files and making the main corpus to work with
    table = pq.read_table('data/Amz_book_review_short.parquet')
    df = table.to_pandas()

    y = df.star_rating.values
    corpus = df.review_body.copy().values
    TextFormating(corpus)() #takes in the corpus and removes stop words, punct, and lems all the words
    
    text_vect = Pipeline([
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer())
    ])
    corpus_vect = text_vect.fit_transform(corpus)
    X_train, X_test, y_train, y_test = train_test_split(corpus_vect, y, test_size=0.33, random_state=42)
    Models = ModelMaker(X_train, y_train)

    # Random Forests
    param_grid_random_forest = {
        'max_depth': [3,None],
        'max_features' : ['sqrt', 'log2', None],
        'min_samples_split': [4,5],
        'min_samples_leaf': [1,2],
        'bootstrap': [True, False],
        'n_estimators': [1,5,10,20,40],
        'random_state': [42]
    }

    RandomForestclf, RandomForestsBestParams = Models.Random_Forest(param_grid_random_forest)
    print(RandomForestsBestParams)
    y_pred = RandomForestclf.predict(X_test)
    RandomForestAcc = np.mean(y_pred == y_test)
    print(RandomForestAcc)

    # Gradient Boosting
    param_grid_grad_boost = {
        'max_depth': [3,4,5,6] ,
        'subsample': [1,.5,.25],
        'max_features' : ['sqrt', 'log2', None],
        'min_samples_split': [4,5],
        'min_samples_leaf': [10,20,30,40],
        'n_estimators': [1,5,10,20,40],
        'random_state': [42]
    }

    GradientBoostclf, GradientBoostBestParams = Models.Grad_Boost(param_grid_grad_boost)
    print(GradientBoostBestParams)
    y_pred = GradientBoostclf.predict(X_test)
    GradientBoostAcc = np.mean(y_pred == y_test)
    print(GradientBoostAcc)