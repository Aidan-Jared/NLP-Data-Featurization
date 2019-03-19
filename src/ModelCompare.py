import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from ModelMaker import ModelMaker
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, mean_squared_error
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from TextCleaning import TextFormating
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import timeit
import spacy

def Get_Corpus(fileName, get_y = True, is_WordVec = False, plot = False):
  df = pq.read_table(fileName).to_pandas()
  if plot:
    barplotdf = df['star_rating'].value_counts()
    barplotdf.plot(kind='bar')
    plt.savefig('images/starting_class_distributions.png')
  if is_WordVec:
    word_vec=[]
    for i in df.review_body.copy():
      word_vec.append(i)
    word_vec = np.vstack(word_vec)
    if get_y:
      y = df.star_rating.values
      return y, word_vec
    return word_vec
  else:
    corpus = df.review_body.copy().values
    if get_y:
      y = df.star_rating.values
      return y, corpus
    return corpus
    
  corpus = df.review_body.copy().values

def scree_plot(ax, pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. variance
    of the projections) for the principal components in a fit sklearn
    PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the scree plot.
      
    title: str
      A title for the scree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), (ind[i]+0.2, vals[i]+0.005), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
	    ax.set_title(title, fontsize=16)

def Smote(X, y, random_state = 42, k_neighbors=3, plot = False):
      smt = SMOTE(random_state=42, k_neighbors=3)
      X_smt, y_smt = smt.fit_sample(X, y)
      if plot:
        df_temp = pd.DataFrame(y_smt, columns=['star_rating'])
        df_temp = df_temp['star_rating'].value_counts()
        df_temp.plot(kind='bar')
        plt.savefig('images/SMOTE_class_distributions.png')
      return X_smt, y_smt

if __name__ == "__main__":
    y, corpus = Get_Corpus('data/Amz_book_review_short.parquet', plot=True)
    word_vec = Get_Corpus('data/Amz_book_review_short_vector.parquet', get_y=False, is_WordVec=True)
    
    text_vect = Pipeline([
                        ('vect', CountVectorizer(max_features=5000, max_df=.85, min_df=2)),
                        ('tfidf', TfidfTransformer())
    ])

    corpus = text_vect.fit_transform(corpus)
    pca = PCA(n_components=300)
    theta = pca.fit_transform(corpus.todense())
    
    fig, ax = plt.subplots(figsize=(8,6))
    scree_plot(ax, pca, title='Test1', n_components_to_plot=20)
    plt.show()

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(theta, y, test_size=0.2, random_state=42, stratify=y)
    X_train_vec, X_test_vec, y_train_vec, y_test_vec = train_test_split(word_vec, y, test_size=0.2, random_state=42, stratify=y)

    X_smt_tfidf, y_smt_tfidf = Smote(X_train_tfidf, y_train_tfidf)
    X_smt_vec, y_smt_vec = Smote(X_train_vec, y_train_vec)

    Models_tfidf = ModelMaker(X_smt_tfidf, y_smt_tfidf)
    Models_vec = ModelMaker(X_smt_vec, y_smt_vec)

    param_grid_random_forest = {
        'max_depth': [None,3,4],
        'max_features' : [None, 'sqrt', 'log2'],
        'min_samples_split': [2,3],
        'min_samples_leaf': [1,2],
        'bootstrap': [True, False],
        'n_estimators': [100,110],
        'random_state': [42]
    }
    # Random_forests_best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 80, 'random_state': 42}

    RandomForestclf_tfidf, RandomForestsBestParams_tfidf = Models_tfidf.Random_Forest(param_grid_random_forest)
    RandomForestclf_vec, RandomForestsBestParams_vec = Models_vec.Random_Forest(param_grid_random_forest)

    print('Params for Random Forest tfidf: ', RandomForestsBestParams_tfidf)
    print('Params for Random Forest Vec: ', RandomForestsBestParams_vec)

    y_pred_random_forest_tfidf = RandomForestclf_tfidf.predict(X_test_tfidf)
    y_pred_random_forest_vec = RandomForestclf_vec.predict(X_test_vec)

    RandomForestMSE_tfidf = mean_squared_error(y_test_tfidf, y_pred_random_forest_tfidf)
    RandomForestMSE_vec = mean_squared_error(y_test_vec, y_pred_random_forest_vec)

    # #Gradient Boosting
    # param_grid_grad_boost = {
    #     'max_depth': [3,4, None],
    #     'subsample': [1,.5],
    #     'max_features' : ['sqrt', 'log2', None],
    #     'min_samples_split': [3,4],
    #     'min_samples_leaf': [5,10],
    #     'n_estimators': [20,30],
    #     'random_state': [42]
    # }
    # # Grad_boost_best_params = {'max_depth': None, 'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 4, 'n_estimators': 20, 'random_state': 42, 'subsample': 1}

    # GradientBoostclf_sent, GradientBoostBestParams_sent = Models_sent.Grad_Boost(param_grid_grad_boost, Plot=False)
    # # GradientBoostclf_vect, GradientBoostBestParams_vect = Models_vect.Grad_Boost(param_grid_grad_boost, Plot=False)
    # # GradientBoostclf_theta, GradientBoostBestParams_theta = Models_theta.Grad_Boost(param_grid_grad_boost, Plot=False)
    # print('Params for Gradient Boost Sent: ', GradientBoostBestParams_sent)
    # # print('Params for Gradient Boost Vect: ', GradientBoostBestParams_vect)
    # # print('Params for Gradient Boost Theta: ', GradientBoostBestParams_theta)
    # y_pred_grad_boost_sent = GradientBoostclf_sent.predict(X_test_sent)
    # y_pred_grad_boost_sent = np.round_(y_pred_grad_boost_sent, 0)
    # # y_pred_grad_boost_vect = GradientBoostclf_vect.predict(X_test_vect)
    # # y_pred_grad_boost_theta = GradientBoostclf_vect.predict(X_test_theta)
    # GradientBoostMSE_sent = mean_squared_error(y_test_sent, y_pred_grad_boost_sent)
    # # GradientBoostMSE_vect = mean_squared_error(y_test_vect, y_pred_grad_boost_vect)
    # # GradientBoostMSE_theta = mean_squared_error(y_test_theta, y_pred_grad_boost_theta)

    # # # Multinomial Naive Bayes
    # # param_grid_Multinomial = {
    # #     'alpha': [.1,.25,.5,.75,1] ,
    # #     'fit_prior': [True, False]
    # # }
    # # # NB_best_params = {'alpha': .1, 'fit_prior': True}

    # # MultinomialNBclf, MultinomialNBBestParams = Models.Naive_Bayes(param_grid_Multinomial, Plot=False)
    # # print(MultinomialNBBestParams)
    # # y_pred_MNB = MultinomialNBclf.predict(X_test)
    # # MultinomialNBMSE = mean_squared_error(y_test, y_pred_MNB)
    # # print(MultinomialNBMSE) # .65875

    # # MLPNN
    # MLP_model_sent, y_pred_MLPNN_sent = Models_sent.MLPNN(X_test_sent, y_test_sent, epoch= 1000, batch_size=10, valaidation_split=.1)
    # # MLP_model_vect, y_pred_MLPNN_vect = Models_vect.MLPNN(X_test_vect, y_test_vect, epoch= 1000, batch_size=10, valaidation_split=.1)
    # # MLP_model_theta, y_pred_MLPNN_theta = Models_theta.MLPNN(X_test_theta, y_test_theta, epoch= 1000, batch_size=10, valaidation_split=.1)
    # y_pred_MLPNN_sent = np.round_(y_pred_MLPNN_sent, 0)
    # MLPNNMSE_sent = mean_squared_error(y_test_sent, y_pred_MLPNN_sent)
    # # MLPNNMSE_vect = mean_squared_error(y_test_vect, y_pred_MLPNN_vect)
    # # MLPNNMSE_theta = mean_squared_error(y_test_theta, y_pred_MLPNN_theta)

    # comparing the models
    modelScores = {'Data Featurization Type': ['Doc2Vec', 'TFIDF'], 'Mean Squared Error' : [RandomForestMSE_vec, RandomForestMSE_tfidf]}
    df_acc = pd.DataFrame.from_dict(modelScores)
    df_acc.plot(kind='bar', x='Data Featurization Type', rot=0)
    plt.savefig('images/Model_MSE.png')