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
    corpus = TextFormating(corpus)() #takes in the corpus and removes stop words, punct, and lems all the words
    
    #vectorizing the text
    # text_vect = Pipeline([
    #                     ('vect', CountVectorizer(max_features=5000, max_df=.85, min_df=2)),
    #                     ('tfidf', TfidfTransformer())
    # ])

    # corpus_vect = text_vect.fit_transform(corpus).todense()
    # # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # # corpus_vect = scaler.fit_transform(corpus_vect)
    # LDA = LatentDirichletAllocation(n_components=5)
    # theta = LDA.fit_transform(corpus_vect)
    
    # # fig, ax = plt.subplots(figsize=(8,6))
    # # scree_plot(ax, pca, title='Test1', n_components_to_plot=20)
    # # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(np.asarray(corpus), y, test_size=0.33, random_state=42, stratify=y)

    #applying SMOTE to Vectors
    smt = SMOTE(random_state=42, k_neighbors=3)
    X_smt, y_smt = smt.fit_sample(X_train, y_train)
    df_temp = pd.DataFrame(y_smt, columns=['star_rating'])
    df_temp = df_temp['star_rating'].value_counts()
    df_temp.plot(kind='bar')
    plt.savefig('images/SMOTE_class_distributions.png')

    Models = ModelMaker(X_smt, y_smt)

    # # Random Forests
    # param_grid_random_forest = {
    #     'max_depth': [None],
    #     'criterion' : ['mse'],
    #     'max_features' : ['sqrt', 'log2', None],
    #     'min_samples_split': [4,5],
    #     'min_samples_leaf': [1],
    #     'bootstrap': [True, False],
    #     'n_estimators': [40,50],
    #     'random_state': [42]
    # }
    # # Random_forests_best_params = {'bootstrap': True, 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 20, 'random_state': 42}

    # RandomForestclf, RandomForestsBestParams = Models.Random_Forest(param_grid_random_forest, Plot=False)
    # print(RandomForestsBestParams)
    # y_pred_random_forest = RandomForestclf.predict(X_test)
    # RandomForestMSE = mean_squared_error(y_test, y_pred_random_forest)
    # print(RandomForestMSE) # .65578

    # #Gradient Boosting
    # param_grid_grad_boost = {
    #     'max_depth': [3,4, None],
    #     'subsample': [1,.5],
    #     'max_features' : ['sqrt', 'log2', None],
    #     'min_samples_split': [4,5],
    #     'min_samples_leaf': [10,20],
    #     'n_estimators': [10,20],
    #     'random_state': [42]
    # }
    # # Grad_boost_best_params = {'max_depth': 3, 'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 4, 'n_estimators': 10, 'random_state': 42, 'subsample': 1}

    # GradientBoostclf, GradientBoostBestParams = Models.Grad_Boost(param_grid_grad_boost, Plot=False)
    # print(GradientBoostBestParams)
    # y_pred_grad_boost = GradientBoostclf.predict(X_test)
    # GradientBoostMSE = mean_squared_error(y_test, y_pred_grad_boost)
    # print(GradientBoostMSE) #.65875

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
    model, y_pred_MLPNN = Models.MLPNN(X_smt, y_smt, epoch= 500, batch_size=10, valaidation_split=.1)
    MLPNNMSE = mean_squared_error(y_test, y_pred_MLPNN)
    print(np.mean((y_test - y_pred_MLPNN)**2))

    # comparing the models
    # modelACC = {'Models': ['Random Forests', 'Gradient Boosting', 'MLP NN'], 'No PCA Accuracy' : [RandomForestMSE, GradientBoostMSE, MLPNNMSE]}
    # df_acc = pd.DataFrame.from_dict(modelACC)
    # df_acc.plot(kind='bar')
    # plt.show()