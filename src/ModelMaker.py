import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping

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

    def Naive_Bayes(self, param_grid, GridSearch = True, Plot =True):
        if GridSearch == True:
            MultinomialNBclf = MultinomialNB()
            search = self.Grid_Search(MultinomialNBclf, param_grid)
            if Plot == True:
                self.plot_results(search)
            return search.best_estimator_, search.best_params_
        else:
            MultinomialNBclf = MultinomialNB(param_grid).fit(self.X,self.y)
            return MultinomialNBclf

    def MLPNN(self, X_test, y_test, epoch, batch_size, valaidation_split):
        y_train_ohe = np_utils.to_categorical(self.y)
        model = self.build_MLPNN(y_train_ohe)
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=40, write_graph=True, write_grads=True, write_images=True)
        earlystop = EarlyStopping(monitor='accuracy', min_delta=.1,patience=0, verbose=0,mode='auto')
        model.fit(self.X, y_train_ohe, epochs=epoch, batch_size=40, verbose=1, validation_split=valaidation_split, callbacks=[tensorboard, earlystop])
        self.print_output(model, X_test, y_test, 42)
        return model
    
    def build_MLPNN(self, y_train_ohe, num_neurons_in_layer = 12, activation = 'tanh', dense = 1):
        model = Sequential()
        num_inputs = self.X.shape[1]
        num_classes = y_train_ohe.shape[1]
        for _ in range(dense):
            model.add(Dense(
                units=num_neurons_in_layer,
                input_dim=num_inputs,
                kernel_initializer='uniform',
                activation= activation
            ))
        model.add(Dense(
                    units=num_classes,
                    input_dim=num_neurons_in_layer,
                    kernel_initializer='uniform',
                    activation='softmax'
                ))
        sgd = SGD(lr=0.001, decay=1e-7, momentum=.9) # learning rate, weight decay, momentum; using stochastic gradient descent (keep)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] )
        return model

    def print_output(self, model, X_test, y_test, rng_seed):
        '''prints model accuracy results'''
        y_train_pred = model.predict_classes(self.X, verbose=0)
        y_test_pred = model.predict_classes(X_test, verbose=0)
        print('\nRandom number generator seed: {}'.format(rng_seed))
        print('\nFirst 30 labels:      {}'.format(self.y[:30]))
        print('First 30 predictions: {}'.format(y_train_pred[:30]))
        train_acc = np.sum(self.y == y_train_pred, axis=0) / self.X.shape[0]
        print('\nTraining accuracy: %.2f%%' % (train_acc * 100))
        test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
        print('Test accuracy: %.2f%%' % (test_acc * 100))
    
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