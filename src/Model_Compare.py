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
#from Word2Vect import Word2Vect
import spacy

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

def WordVect(X):
    for idx, i in enumerate(X):
        doc = nlp(i)
        X[idx] = doc.vector
    return np.vstack(X)

if __name__ == "__main__":
    #importing files and making the main corpus to work with
    nlp = spacy.load('en_core_web_lg')
    table = pq.read_table('data/Amz_book_review_short.parquet')
    df = table.to_pandas()
    
    #showing the distribution of values
    barplotthing = df['star_rating'].value_counts()
    barplotthing.plot(kind='bar')
    plt.savefig('images/starting_class_distributions.png')

    #formating the text
    y = df.star_rating.values
    corpus = df.review_body.copy().values
    corpus_sent = TextFormating(corpus, Sentiment=False)() #takes in the corpus and removes stop words, punct, and lems all the words
    # corpus_p2v = Word2Vect(corpus_vect)()
    #corpus_sent = np.asarray(corpus_sent)
    corpus_sent = WordVect(corpus_sent)
    
    # #vectorizing the text
    # text_vect = Pipeline([
    #                     ('vect', CountVectorizer(max_features=5000, max_df=.85, min_df=2)),
    #                     ('tfidf', TfidfTransformer())
    # ])

    # corpus_vect = text_vect.fit_transform(corpus_vect).todense()
    # LDA = LatentDirichletAllocation(n_components=5)
    # theta = LDA.fit_transform(corpus_vect)
    
    # # fig, ax = plt.subplots(figsize=(8,6))
    # # scree_plot(ax, pca, title='Test1', n_components_to_plot=20)
    # # plt.show()

    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(corpus_sent, y, test_size=0.33, random_state=42, stratify=y)
    # X_train_vect, X_test_vect, y_train_vect, y_test_vect = train_test_split(corpus_vect, y, test_size=0.33, random_state=42, stratify=y)
    # X_train_theta, X_test_theta, y_train_theta, y_test_theta = train_test_split(theta, y, test_size=0.33, random_state=42, stratify=y)

    #applying SMOTE to Vectors
    X_smt_sent, y_smt_sent = Smote(X_train_sent, y_train_sent)
    # X_smt_vect, y_smt_vect = Smote(X_train_vect, y_train_vect)
    # X_smt_theta, y_smt_theta = Smote(X_train_theta, y_train_theta)

    Models_sent = ModelMaker(X_smt_sent, y_smt_sent)
    # Models_vect = ModelMaker(X_smt_vect, y_smt_vect)
    # Models_theta = ModelMaker(X_smt_theta, y_smt_theta)

    # Random Forests
    param_grid_random_forest = {
        'max_depth': [None,3],
        'max_features' : ['sqrt', 'log2'],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True, False],
        'n_estimators': [80,90,100],
        'random_state': [42]
    }
    # Random_forests_best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 80, 'random_state': 42}

    RandomForestclf_sent, RandomForestsBestParams_sent = Models_sent.Random_Forest(param_grid_random_forest, Plot=False)
    # RandomForestclf_vect, RandomForestsBestParams_vect = Models_vect.Random_Forest(param_grid_random_forest, Plot=False)
    # RandomForestclf_theta, RandomForestsBestParams_theta = Models_theta.Random_Forest(param_grid_random_forest, Plot=False)
    print('Params for Random Forest Sent: ', RandomForestsBestParams_sent)
    # print('Params for Random Forest Vect: ', RandomForestsBestParams_vect)
    # print('Params for Random Forest Theta: ', RandomForestsBestParams_theta)
    y_pred_random_forest_sent = RandomForestclf_sent.predict(X_test_sent)
    y_pred_random_forest_sent = np.round_(y_pred_random_forest_sent, 0)
    # y_pred_random_forest_vect = RandomForestclf_vect.predict(X_test_vect)
    # y_pred_random_forest_theta = RandomForestclf_vect.predict(X_test_theta)
    RandomForestMSE_sent = mean_squared_error(y_test_sent, y_pred_random_forest_sent)
    # RandomForestMSE_vect = mean_squared_error(y_test_vect, y_pred_random_forest_vect)
    # RandomForestMSE_theta = mean_squared_error(y_test_vect, y_pred_random_forest_theta)

    #Gradient Boosting
    param_grid_grad_boost = {
        'max_depth': [3,4, None],
        'subsample': [1,.5],
        'max_features' : ['sqrt', 'log2', None],
        'min_samples_split': [3,4],
        'min_samples_leaf': [5,10],
        'n_estimators': [20,30],
        'random_state': [42]
    }
    # Grad_boost_best_params = {'max_depth': None, 'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 4, 'n_estimators': 20, 'random_state': 42, 'subsample': 1}

    GradientBoostclf_sent, GradientBoostBestParams_sent = Models_sent.Grad_Boost(param_grid_grad_boost, Plot=False)
    # GradientBoostclf_vect, GradientBoostBestParams_vect = Models_vect.Grad_Boost(param_grid_grad_boost, Plot=False)
    # GradientBoostclf_theta, GradientBoostBestParams_theta = Models_theta.Grad_Boost(param_grid_grad_boost, Plot=False)
    print('Params for Gradient Boost Sent: ', GradientBoostBestParams_sent)
    # print('Params for Gradient Boost Vect: ', GradientBoostBestParams_vect)
    # print('Params for Gradient Boost Theta: ', GradientBoostBestParams_theta)
    y_pred_grad_boost_sent = GradientBoostclf_sent.predict(X_test_sent)
    y_pred_grad_boost_sent = np.round_(y_pred_grad_boost_sent, 0)
    # y_pred_grad_boost_vect = GradientBoostclf_vect.predict(X_test_vect)
    # y_pred_grad_boost_theta = GradientBoostclf_vect.predict(X_test_theta)
    GradientBoostMSE_sent = mean_squared_error(y_test_sent, y_pred_grad_boost_sent)
    # GradientBoostMSE_vect = mean_squared_error(y_test_vect, y_pred_grad_boost_vect)
    # GradientBoostMSE_theta = mean_squared_error(y_test_theta, y_pred_grad_boost_theta)

    # # Multinomial Naive Bayes
    # param_grid_Multinomial = {
    #     'alpha': [.1,.25,.5,.75,1] ,
    #     'fit_prior': [True, False]
    # }
    # # NB_best_params = {'alpha': .1, 'fit_prior': True}

    # MultinomialNBclf, MultinomialNBBestParams = Models.Naive_Bayes(param_grid_Multinomial, Plot=False)
    # print(MultinomialNBBestParams)
    # y_pred_MNB = MultinomialNBclf.predict(X_test)
    # MultinomialNBMSE = mean_squared_error(y_test, y_pred_MNB)
    # print(MultinomialNBMSE) # .65875

    # MLPNN
    MLP_model_sent, y_pred_MLPNN_sent = Models_sent.MLPNN(X_test_sent, y_test_sent, epoch= 1000, batch_size=10, valaidation_split=.1)
    # MLP_model_vect, y_pred_MLPNN_vect = Models_vect.MLPNN(X_test_vect, y_test_vect, epoch= 1000, batch_size=10, valaidation_split=.1)
    # MLP_model_theta, y_pred_MLPNN_theta = Models_theta.MLPNN(X_test_theta, y_test_theta, epoch= 1000, batch_size=10, valaidation_split=.1)
    y_pred_MLPNN_sent = np.round_(y_pred_MLPNN_sent, 0)
    MLPNNMSE_sent = mean_squared_error(y_test_sent, y_pred_MLPNN_sent)
    # MLPNNMSE_vect = mean_squared_error(y_test_vect, y_pred_MLPNN_vect)
    # MLPNNMSE_theta = mean_squared_error(y_test_theta, y_pred_MLPNN_theta)

    # comparing the models
    modelACC = {'Models': ['Random Forests', 'Gradient Boosting', 'MLP NN'], 'Sentement_MSE' : [RandomForestMSE_sent, GradientBoostMSE_sent, MLPNNMSE_sent]}
    df_acc = pd.DataFrame.from_dict(modelACC)
    df_acc.plot(kind='bar', x='Models', rot=0)
    plt.savefig('images/Model_MSE_new.png')