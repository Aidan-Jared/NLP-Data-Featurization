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
    
    def Random_Forest(self, param_grid, GridSearch = True):
        if GridSearch == True:
            
            RandomForestReg = RandomForestRegressor()
            search = self.Grid_Search(RandomForestReg, param_grid)
            return search.best_estimator_, search.best_params_
        else:
            bootstrap = param_grid['bootstrap']
            max_features = param_grid['max_features']
            max_depth = param_grid['max_depth']
            min_samples_leaf = param_grid['min_samples_leaf']
            min_samples_split = param_grid['min_samples_split']
            n_estimators = param_grid['n_estimators']
            RandomForestReg = RandomForestRegressor(n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=1).fit(self.X,self.y)
            return RandomForestReg, param_grid

    def Grad_Boost(self, param_grid, GridSearch = True):
        if GridSearch == True:
            GradientBoostingReg = GradientBoostingRegressor()
            search = self.Grid_Search(GradientBoostingReg, param_grid)
            return search.best_estimator_, search.best_params_
        else:
            GradientBoostingReg = GradientBoostingRegressor(param_grid).fit(self.X,self.y)
            return GradientBoostingReg

    def MLPNN(self, X_test, y_test, epoch = 200, batch_size = 40, valaidation_split= .1):
        model = self.build_MLPNN()
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=40, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='loss', min_delta=1e-5,patience=30, verbose=0,mode='auto')
        model.fit(self.X, self.y, epochs=epoch, batch_size=batch_size, verbose=1, validation_split=valaidation_split, callbacks=[tensorboard, earlystop])
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

    def Grid_Search(self, model, param_grid):
        search = GridSearchCV(model, 
                            param_grid,
                            n_jobs=-1, verbose=1, cv=3,
                            scoring='neg_mean_squared_error')
        search.fit(self.X, self.y)
        return search