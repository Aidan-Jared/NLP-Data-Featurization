import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard, EarlyStopping

class ModelMaker(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.weights = [
                {5:1, 4:10},
                {5:1, 3:10},
                {5:1, 1:10},
                {5:1, 2:10}
            ]
    
    def Random_Forest(self, param_grid, GridSearch = True, Plot =True):
        if GridSearch == True:
            
            RandomForestReg = RandomForestRegressor()
            search = self.Grid_Search(RandomForestReg, param_grid)
            # if Plot == True:
            #     self.plot_results(search)
            return search.best_estimator_, search.best_params_
        else:
            RandomForestReg = RandomForestRegressor(param_grid).fit(self.X,self.y)
            return RandomForestReg

    def Grad_Boost(self, param_grid, GridSearch = True, Plot =True):
        if GridSearch == True:
            GradientBoostingReg = GradientBoostingRegressor()
            search = self.Grid_Search(GradientBoostingReg, param_grid)
            # if Plot == True:
            #     self.plot_results(search)
            return search.best_estimator_, search.best_params_
        else:
            GradientBoostingReg = GradientBoostingRegressor(param_grid).fit(self.X,self.y,sample_weight=self.weights)
            return GradientBoostingReg

    # def Naive_Bayes(self, param_grid, GridSearch = True, Plot =True):
    #     if GridSearch == True:
    #         MultinomialNBclf = MultinomialNB()
    #         search = self.Grid_Search(MultinomialNBclf, param_grid)
    #         # if Plot == True:
    #         #     self.plot_results(search)
    #         return search.best_estimator_, search.best_params_
    #     else:
    #         MultinomialNBclf = MultinomialNB(param_grid).fit(self.X,self.y)
    #         return MultinomialNBclf

    def MLPNN(self, X_test, y_test, epoch = 200, batch_size = 40, valaidation_split= .1):
        model = self.build_MLPNN()
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=40, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='loss', min_delta=1e-5,patience=30, verbose=0,mode='auto')
        model.fit(self.X, self.y, epochs=epoch, batch_size=batch_size, verbose=1, validation_split=valaidation_split, callbacks=[tensorboard, earlystop]) #
        self.print_output(model, X_test, y_test, 42)
        y_test_pred = model.predict(X_test, verbose=0)
        return model, y_test_pred.T[0]
    
    def build_MLPNN(self, num_neurons_in_layer = 100, activation = 'linear', dense = 1):
        model = Sequential()
        num_inputs = self.X.shape[1]
        model.add(Dense(
                units=4,
                input_dim=num_inputs,
                kernel_initializer='uniform',
                activation= activation
        ))
        model.add(Dense(
            units=num_neurons_in_layer,
            activation = activation
        ))

        model.add(Dropout(.25))
        
        model.add(Dense(
                    units=1,
                    input_dim=num_neurons_in_layer,
                    kernel_initializer='uniform',
                    activation='linear'
                ))

        ada = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=ada)
        return model

    def print_output(self, model, X_test, y_test, rng_seed):
        '''prints model accuracy results'''
        y_train_pred = model.predict(self.X, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        y_test_pred = y_test_pred.T[0]
        print('\nRandom number generator seed: {}'.format(rng_seed))
        print('\nFirst 30 labels:      {}'.format(y_test[:30]))
        print('First 30 predictions: {}'.format(y_test_pred[:30]))
        train_MSE = mean_squared_error(self.y, y_train_pred)
        print('\nTraining MSE: %.2f' % (train_MSE))
        test_MSE = mean_squared_error(y_test, y_test_pred)
        print('Test MSE: %.2f' % (test_MSE))
    
    def Grid_Search(self, model, param_grid):
        search = GridSearchCV(model, 
                            param_grid,
                            n_jobs=-1, verbose=1, cv=3,
                            scoring='neg_mean_squared_error')
        search.fit(self.X, self.y)
        return search

    # def plot_results(self, model, param = 'n_estimators', name = 'Num Trees'):
    #     param_name = 'param_%s' % param

    #     # Extract information from the cross validation model
    #     train_scores = model.cv_results_['mean_train_score']
    #     test_scores = model.cv_results_['mean_test_score']
    #     train_time = model.cv_results_['mean_fit_time']
    #     param_values = list(model.cv_results_[param_name])
        
    #     # Plot the scores over the parameter
    #     plt.subplots(1, 2, figsize=(10, 6))
    #     plt.subplot(121)
    #     plt.plot(param_values, train_scores, 'bo-', label = 'train')
    #     plt.plot(param_values, test_scores, 'go-', label = 'test')
    #     plt.ylim(ymin = -10, ymax = 0)
    #     plt.legend()
    #     plt.xlabel(name)
    #     plt.ylabel('Neg Mean Absolute Error')
    #     plt.title('Score vs %s' % name)
        
    #     plt.subplot(122)
    #     plt.plot(param_values, train_time, 'ro-')
    #     plt.ylim(ymin = 0.0, ymax = 2.0)
    #     plt.xlabel(name)
    #     plt.ylabel('Train Time (sec)')
    #     plt.title('Training Time vs %s' % name)
        
        
    #     plt.tight_layout(pad = 4)
    #     plt.show()