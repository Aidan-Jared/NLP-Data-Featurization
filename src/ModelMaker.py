import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline

class ModelMaker(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def Random_Forest(self, param_grid, GridSearch = True, Plot =True):
        if GridSearch == True:
            RandomForestclf = RandomForestClassifier()
            search = self.Grid_Search(RandomForestclf, param_grid)
            if Plot == True:
                self.plot_results(search)
            return search.best_estimator_, search.best_params_
        else:
            RandomForestclf = RandomForestClassifier(param_grid).fit(self.X,self.y)
            return RandomForestclf

    def Grad_Boost(self, param_grid, GridSearch = True, Plot =True):
        if GridSearch == True:
            GradientBoostingclf = GradientBoostingClassifier()
            search = self.Grid_Search(GradientBoostingclf, param_grid)
            if Plot == True:
                self.plot_results(search)
            return search.best_estimator_, search.best_params_
        else:
            GradientBoostingclf = GradientBoostingClassifier(param_grid).fit(self.X,self.y)
            return GradientBoostingclf

    def Grid_Search(self, model, param_grid):
        search = GridSearchCV(model, 
                            param_grid,
                            n_jobs=-1, verbose=1, cv=3,
                            scoring='neg_mean_squared_error')
        search.fit(self.X, self.y)
        return search

    def plot_results(self, model, param = 'n_estimators', name = 'Num Trees'):
        param_name = 'param_%s' % param

        # Extract information from the cross validation model
        train_scores = model.cv_results_['mean_train_score']
        test_scores = model.cv_results_['mean_test_score']
        train_time = model.cv_results_['mean_fit_time']
        param_values = list(model.cv_results_[param_name])
        
        # Plot the scores over the parameter
        plt.subplots(1, 2, figsize=(10, 6))
        plt.subplot(121)
        plt.plot(param_values, train_scores, 'bo-', label = 'train')
        plt.plot(param_values, test_scores, 'go-', label = 'test')
        plt.ylim(ymin = -10, ymax = 0)
        plt.legend()
        plt.xlabel(name)
        plt.ylabel('Neg Mean Absolute Error')
        plt.title('Score vs %s' % name)
        
        plt.subplot(122)
        plt.plot(param_values, train_time, 'ro-')
        plt.ylim(ymin = 0.0, ymax = 2.0)
        plt.xlabel(name)
        plt.ylabel('Train Time (sec)')
        plt.title('Training Time vs %s' % name)
        
        
        plt.tight_layout(pad = 4)
        plt.show()