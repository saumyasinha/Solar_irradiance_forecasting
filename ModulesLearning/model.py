import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# import xgboost as xgb
from random import seed
from random import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, sgd
import matplotlib.pyplot as plt


def lr_model(X, y):
    reg = LinearRegression().fit(X, y)
    return reg


def lasso_model(X, y):
    reg = LassoCV().fit(X, y)
    return reg


def rf_model(X, y):
    reg = RandomForestRegressor()  # max_depth=5, random_state=0)
    reg.fit(X, y)
    return reg


def rfSearch_model(X, y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)
    return rf_random


def rfGridSearch_model(X, y):

    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 100, 120],
        'max_features': [3, 5],
        'min_samples_leaf': [8, 10, 15],
        'min_samples_split': [8, 10, 15],
        'n_estimators': [100, 150, 200, 250]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print("params:", grid_search.best_params_)
    return grid_search


# neural network based model........
def FNN_model(X_train, y_train, bs, epochs):
    # Model Structure
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #     model.add(Dense(5, activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #     model.add(Dropout(0.5))

    model.add(Dense(5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))

    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])
    model.summary()
    history = model.fit(X_train, y_train, batch_size=bs, epochs=epochs, verbose=2)
    return [model, history]