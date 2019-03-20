import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from ModelMaker import ModelMaker
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.pipeline import Pipeline
from TextCleaning import TextFormating
from Doc2Vect import Doc2Vect
from imblearn.over_sampling import SMOTE
from timeit import default_timer as timer
import spacy
plt.style.use('ggplot')

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

def scree_plot(ax, pca, n_components_to_plot=8, title='PCA Scree Plot'):
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

def Doc_Simularity(corpus, vector, index, n_simular = 1):
    cosine = pairwise_distances(vector, metric='cosine')[index]
    indexes_most = cosine.argsort()[:-n_simular-1:-1][0]
    indexes_least = cosine.argsort()[n_simular+1]
    return corpus[indexes_most], corpus[indexes_least]

def ModelSplitting(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=5, stratify=y)
    return X_train, X_test, y_train, y_test

def ModelEvaluation(model, params, X_test, y_test):
    start_build_model = timer()
    Bestmodel, BestParams = model.Random_Forest(params, GridSearch=False)
    end_build_model = timer()

    start_preidct = timer()
    y_pred = Bestmodel.predict(X_test)
    y_pred = np.round(y_pred)
    end_preidct = timer()

    build_time = end_build_model - start_build_model
    predict_time = end_preidct - start_preidct

    modelMSE = mean_squared_error(y_test, y_pred)
    return Bestmodel, BestParams, y_pred, modelMSE, build_time, predict_time

def bar_plot(Dict, y_label, title, save_file, legend=False):
    df = pd.DataFrame.from_dict(Dict)
    ax = df.plot(kind='bar', x='Data Featurization Type', rot=0, legend=legend)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for i in ax.patches:
      ax.annotate(str(i.get_height())[:5], (i.get_x(), i.get_height() * 1.01))
    plt.savefig(save_file)

def Document_Save_to_file(filename, corpus, word_vec):
      for index in np.random.randint(0,len(corpus),5):
        most_simular, least_simular = Doc_Simularity(corpus, word_vec, index)
        with open(filename, 'a') as f:
          f.write("New Doc: ")
          f.write('\n')
          f.write(corpus[index])
          f.write('\n')
          f.write(most_simular)
          f.write('\n')
          f.write(least_simular)
          f.write('\n')

if __name__ == "__main__":
    y, corpus = Get_Corpus('data/Amz_book_review_short.parquet', plot=False)
    word_vec = Get_Corpus('data/Amz_book_review_short_vector.parquet', get_y=False, is_WordVec=True)
    
    # corpus_big, corpus_short, y_big, y_short = ModelSplitting(corpus, y, .25)
    # word_vec, word_vec_short, y_big, y_short = ModelSplitting(word_vec, y, .25)
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(corpus, y, test_size=.2, random_state=42, stratify=y)
    X_train_vec_spacy, X_test_vec_spacy, y_train_vec_spacy, y_test_vec_spacy = train_test_split(word_vec, y, test_size=.2, random_state=42, stratify=y)
    X_train_vec_gensim, X_test_vec_gensim = Doc2Vect(X_train_tfidf, X_test_tfidf)()
    
    Document_Save_to_file('DocVecspacy.txt', corpus, word_vec)
    Document_Save_to_file('DocVecgensim.txt', X_train_tfidf, X_train_vec_gensim)
    
    text_vect = Pipeline([
                        ('vect', CountVectorizer(max_features=5000, max_df=.85, min_df=2)),
                        ('tfidf', TfidfTransformer())
    ])

    X_train_tfidf = text_vect.fit_transform(X_train_tfidf).todense()
    X_test_tfidf = text_vect.transform(X_test_tfidf)
    pca = PCA(n_components=100)
    theta = pca.fit_transform(X_train_tfidf)
    
    # fig, ax = plt.subplots(figsize=(8,6))
    # scree_plot(ax, pca, n_components_to_plot=5000)
    # plt.savefig('images/Screeplot.png')

    X_smt_tfidf, y_smt_tfidf = Smote(X_train_tfidf, y_train_tfidf)
    X_smt_vec_spacy, y_smt_vec_spacy = Smote(X_train_vec_spacy, y_train_vec_spacy)
    X_smt_vec_gensim, y_smt_vec_gensim = Smote(X_train_vec_gensim, y_train_tfidf)

    Models_tfidf = ModelMaker(X_smt_tfidf, y_smt_tfidf)
    Models_vec_spacy = ModelMaker(X_smt_vec_spacy, y_smt_vec_spacy)
    Models_vec_gensim = ModelMaker(X_smt_vec_gensim, y_smt_vec_gensim)

    # param_grid_random_forest = {
    #     'max_depth': [None,3],
    #     'max_features' : ['sqrt', 'log2'],
    #     'min_samples_split': [2,3,4],
    #     'min_samples_leaf': [1,2],
    #     'bootstrap': [True, False],
    #     'n_estimators': [80,90,100],
    #     'random_state': [42]
    # }
    Params_Random_Forest_tfidf =   {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 80, 'random_state': 42}
    Params_Random_Forest_Spacy_Vec =   {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 90, 'random_state': 42}
    Params_Random_Forest_Gensim_Vec =  {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 80, 'random_state': 42}

    RandomForestclf_tfidf, RandomForestsBestParams_tfidf, y_pred_random_forest_tfidf, RandomForestMSE_tfidf, tfidf_build_time, tfidf_predict_time = ModelEvaluation(Models_tfidf, Params_Random_Forest_tfidf, X_test_tfidf, y_test_tfidf)
    RandomForestclf_vec_spacy, RandomForestsBestParams_vec_spacy, y_pred_random_forest_vec_spacy, RandomForestMSE_vec_spacy, spacy_build_time, spacy_predict_time = ModelEvaluation(Models_vec_spacy, Params_Random_Forest_Spacy_Vec, X_test_vec_spacy, y_test_vec_spacy)
    RandomForestclf_vec_gensim, RandomForestsBestParams_vec_gensim, y_pred_random_forest_vec_gensim, RandomForestMSE_vec_gensim, gensim_build_time, gensim_predict_time = ModelEvaluation(Models_vec_gensim, Params_Random_Forest_Gensim_Vec, X_test_vec_gensim, y_test_tfidf)

    print('Params for Random Forest tfidf: ', RandomForestsBestParams_tfidf)
    print('Params for Random Forest Spacy Vec: ', RandomForestsBestParams_vec_spacy)
    print('Params for Random Forest Gensim Vec: ', RandomForestsBestParams_vec_gensim)

    # comparing the models
    modelScores = {'Data Featurization Type': ['Gensim Doc2Vec','Spacy Doc2Vec', 'TFIDF'], 'Mean Squared Error' : [RandomForestMSE_vec_gensim, RandomForestMSE_vec_spacy, RandomForestMSE_tfidf]}
    modelTime = {'Data Featurization Type': ['Gensim Doc2Vec','Spacy Doc2Vec', 'TFIDF'], 'Time To Build Model' : [gensim_build_time, spacy_build_time, tfidf_build_time], 'Time to Predict' : [gensim_predict_time, spacy_predict_time, tfidf_predict_time]}

    bar_plot(modelScores, 'Mean Squared Error', 'MSE for Data Featurization', 'images/Model_MSE.png')
    bar_plot(modelTime, 'Time (Seconds)', 'Model Time for Data Featurization', 'images/Model_Time.png', legend=True)