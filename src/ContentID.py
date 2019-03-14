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
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    #importing files and making the main corpus to work with
    table = pq.read_table('data/Amz_book_review_short.parquet')
    df = table.to_pandas()
    
    #showing the distribution of values
    barplotthing = df['star_rating'].value_counts()
    barplotthing.plot(kind='bar')
    plt.savefig('images/starting_class_distributions.png')

    #formating the text
    y = df.star_rating.values
    corpus = df.review_body.copy().values
    TextFormating(corpus)() #takes in the corpus and removes stop words, punct, and lems all the words
    
    #vectorizing the text
    text_vect = Pipeline([
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer())
    ])
    corpus_vect = text_vect.fit_transform(corpus).todense()
    X_train, X_test, y_train, y_test = train_test_split(corpus_vect, y, test_size=0.33, random_state=42, stratify=y)

    #applying SMOTE to Vectors
    smt = SMOTE(random_state=42, k_neighbors=1)
    X_smt, y_smt = smt.fit_sample(X_train, y_train)
    df_temp = pd.DataFrame(y_smt, columns=['star_rating'])
    df_temp = df_temp['star_rating'].value_counts()
    df_temp.plot(kind='bar')
    plt.savefig('images/SMOTE_class_distributions.png')

    Models = ModelMaker(X_train, y_train)

    # # Random Forests
    # param_grid_random_forest = {
    #     'max_depth': [None],
    #     'max_features' : ['sqrt', 'log2', None],
    #     'min_samples_split': [3,4],
    #     'min_samples_leaf': [1,2],
    #     'bootstrap': [True],
    #     'n_estimators': [20,30,40],
    #     'random_state': [42]
    # }
    # # Random_forests_best_params = {'bootstrap': True, 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 20, 'random_state': 42}

    # RandomForestclf, RandomForestsBestParams = Models.Random_Forest(param_grid_random_forest, Plot=False)
    # print(RandomForestsBestParams)
    # y_pred_random_forest = RandomForestclf.predict(X_test)
    # RandomForestAcc = np.mean(y_pred_random_forest == y_test)
    # print(RandomForestAcc) # .65578

    # # Gradient Boosting
    # param_grid_grad_boost = {
    #     'max_depth': [3,4, None] ,
    #     'subsample': [1,.5],
    #     'max_features' : ['sqrt', 'log2', None],
    #     'min_samples_split': [4,5],
    #     'min_samples_leaf': [10,20],
    #     'n_estimators': [10,20],
    #     'random_state': [42]
    # }
    # # Grad_boost_best_params = 

    # GradientBoostclf, GradientBoostBestParams = Models.Grad_Boost(param_grid_grad_boost, Plot=False)
    # print(GradientBoostBestParams)
    # y_pred_grad_boost = GradientBoostclf.predict(X_test)
    # GradientBoostAcc = np.mean(y_pred_grad_boost == y_test)
    # print(GradientBoostAcc)

    # Multinomial Naive Bayes
    # param_grid_Multinomial = {
    #     'alpha': [0,.25,.5,.75,1] ,
    #     'fit_prior': [True, False]
    # }
    # # NB_best_params = {'alpha': .1, 'fit_prior': True}

    # MultinomialNBclf, MultinomialNBBestParams = Models.Naive_Bayes(param_grid_Multinomial, Plot=False)
    # print(MultinomialNBBestParams)
    # y_pred_MNB = MultinomialNBclf.predict(X_test)
    # MultinomialNBAcc = np.mean(y_pred_MNB == y_test)
    # print(MultinomialNBAcc) # .6097

    # MLPNN
    model = Models.MLPNN(X_test, y_test, epoch= 100, batch_size=10, valaidation_split=.1)