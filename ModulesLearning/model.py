import numpy as np
import time
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from random import seed
from random import random
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.svm import LinearSVR
from matplotlib import pyplot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import skorch
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV
from torch.autograd import Variable
from SolarForecasting.ModulesLearning import preprocessing as preprocess
from keras.models import Sequential
from keras.layers import RNN,Dense, LSTM, Dropout,Bidirectional,RepeatVector,TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.models import load_model



class FFNN(nn.Module):
    """Simple FF network with multiple outputs.
    """
    def __init__(
        self,
        input_size,
        hidden_sizes,
        dropout_rate=.5,
    ):
        """
        :param input_size: input size
        :param hidden_size: common hidden size for all layers
        :param n_hidden: number of hidden layers
        :param n_outputs: number of outputs
        :param dropout_rate: dropout rate
        """
        super().__init__()
        assert 0 <= dropout_rate < 1
        self.input_size = input_size

        h_sizes = [self.input_size] + hidden_sizes
        self.h_sizes = h_sizes
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(
                nn.Linear(
                    h_sizes[k],
                    h_sizes[k + 1]
                )
            )

        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        for layer in self.hidden[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        return self.hidden[-1](x)


class Network(nn.Module):

    def __init__(
            self,
            input_size = 36,
            hidden_sizes=[32,32],
            dropout_rate=.5,
    ):
        super().__init__()

    #     self.l1 = nn.Linear(input_size, 32)
    #     self.l2 = nn.Linear(32, 32)
    #     self.l3 = nn.Linear(32,1)
    #     # self.l4 = nn.Linear(8,1)
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(0.5)
    #
    # def forward(self, x):
    #     x = self.l1(x)
    #     x = self.dropout(self.relu(x))
    #     x = self.l2(x)
    #     x = self.dropout(self.relu(x))
    #     x = self.l3(x)
    #     # x = self.dropout(self.relu(x))
    #     # x = self.l4(x)
    #     return x

        self.model = nn.Sequential()

        self.model.add_module(
            'hard_sharing',
            FFNN(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate
            )
        )

        # self.model.add_module('relu1', nn.ReLU())
        # self.model.add_module('dropout1', nn.Dropout(p=dropout_rate))
        # self.model.add_module('fc1', nn.Linear(hidden_sizes[-1], 16))
        self.model.add_module('relu', nn.ReLU())
        self.model.add_module('dropout', nn.Dropout(p=dropout_rate))
        self.model.add_module('fc', nn.Linear(hidden_sizes[-1], 1))


    def forward(self,x):
        return self.model(x)




def fnn_train(X_train, y_train, folder_saving, epochs=500, model_saved="FNN"):

    input_size = X_train.shape[1]
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    net = NeuralNetRegressor(
        Network,
        criterion=nn.MSELoss,
        max_epochs=epochs,
        optimizer=optim.Adam,
        batch_size=64,
        optimizer__lr=.001,
        optimizer__weight_decay = 1e-5
    )
    print(net)

    params = {
        'optimizer__lr': [0.01, 0.001, 0.0001],
        'module__dropout_rate' : [0.5,0],
        'module__hidden_sizes': [[64,16],[128,64],[32,32],[24,16], [64,16,8]]
    }
    #
    gs = RandomizedSearchCV(net, param_distributions=params, refit=True,cv=3,scoring='neg_mean_squared_error',n_iter=100)
    gs.fit(X_train, y_train)
    print("params:", gs.best_params_)
    # for lead 5: params: {'optimizer__lr': 0.001, 'module__hidden_sizes': [64, 16], 'batch_size': 32}
    # for lead 6: params: {'optimizer__lr': 0.001, 'module__hidden_sizes': [64, 16], 'batch_size': 32}
    # net.fit(X_train,y_train)
    gs.best_estimator_.save_params(f_params= folder_saving + model_saved+".pkl")

    epochs = [i for i in range(len(gs.best_estimator_.history))]
    train_loss = gs.best_estimator_.history[:, 'train_loss']
    valid_loss = gs.best_estimator_.history[:, 'valid_loss']
    plt.plot(epochs, train_loss, 'g-')
    plt.plot(epochs, valid_loss, 'r-')
    plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend(['Train', 'Validation'])
    plt.savefig(folder_saving+"loss_plots_"+model_saved)

    return gs
    # return gs.best_estimator_

    # net = Network(input_size=input_size,
    #             hidden_sizes=hidden_sizes)
    # print(net)

    # train_loss_list = []
    # valid_loss_list = []
    #
    # count_train, count_valid = X_train.shape[0], X_valid.shape[0]
    # print(count_train, count_valid)
    #
    # X_train = torch.from_numpy(X_train)
    # X_train = X_train.float()
    # y_train = torch.tensor(y_train)
    # y_train = y_train.float()
    #
    # X_valid = torch.from_numpy(X_valid)
    # X_valid = X_valid.float()
    # y_valid = torch.tensor(y_valid)
    # y_valid = y_valid.float()
    #
    # optimizer = optim.Adam(net.parameters(), weight_decay=0.0005)
    # decayRate = 0.96
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    #
    # loss_func = nn.MSELoss()
    #
    # valid_loss_min = np.Inf
    #
    # for epoch in range(epochs):
    #     minibatches = random_mini_batches(X_train, y_train, batch_size)
    #     train_loss = 0
    #
    #     my_lr_scheduler.step()
    #     net.train()
    #     for x_mini, y_mini in minibatches:
    #         # x_mini = X_train[i:i + batch_size]
    #         # y_mini = y_train[i:i + batch_size]
    #
    #         optimizer.zero_grad()
    #         net_out = net(x_mini)
    #
    #         loss = loss_func(net_out, y_mini)
    #         loss.backward()
    #         optimizer.step()
    #
    #         train_loss += loss.item()*x_mini.size(0)
    #
    #     train_loss_list.append(train_loss/count_train)
    #
    #     net.eval()
    #
    #     y_pred = net(X_valid)
    #     valid_loss = loss_func(y_pred,y_valid).item()
    #     valid_loss_list.append(valid_loss)
    #
    #     # print training/validation statistics
    #     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    #         epoch, train_loss/count_train, valid_loss))
    #
    #     # save model if validation loss has decreased
    #     if valid_loss <= valid_loss_min:
    #         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    #             valid_loss_min,
    #             valid_loss))
    #         torch.save(net.state_dict(), folder_saving + model_saved)
    #         valid_loss_min = valid_loss
    #
    # loss_plots(train_loss_list, valid_loss_list, folder_saving, loss_type=model_saved)
    #
    # return net


def fnn_test(X_test, net, model_path=None):

    net_out = net.predict(X_test.astype(np.float32))

    return net_out



###### Ensemble models ##############
def rfSearch_model(X, y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=800, num=7)]
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
    # bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,scoring = 'neg_mean_squared_error', n_iter=100, cv=3,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)
    return rf_random

def rfGridSearch_model(X, y):

    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 100, 120],
        'max_features': [3, 5, 7],
        'min_samples_leaf': [8, 10, 15],
        'min_samples_split': [8, 10, 15],
        'n_estimators': [100, 200, 300]

    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, scoring = 'neg_mean_squared_error')
    grid_search.fit(X, y)
    print("params:", grid_search.best_params_)
    return grid_search

# def xg_boost(X_train, y_train):
#     param_grid = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#     }
#
#     xgb = XGBRegressor()
#
#     grid_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train, y_train)
#     print("params:", grid_search.best_params_)
#     return grid_search


### LSTM/RNN method
def lstm_model(X_train, y_train, folder_saving, model_saved,timesteps=1, n_features = 1, n_outputs = 1):


    ##why giving nans??
    print("before: ",X_train.shape)
    X_train = X_train.reshape((X_train.shape[0], timesteps,n_features))
    # y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    print("after: ",X_train.shape)

    epochs_ = 150
    batch_size_ = 32
    dropout_ = .3

    # design network                                                            #
    model = Sequential()  #
    model.add(LSTM(15,activation="relu",input_shape=(X_train.shape[1], X_train.shape[2])))  #
    model.add(Dropout(dropout_))  #
    model.add(Dense(n_outputs))  #
    model.compile(loss='mean_squared_error', optimizer='adam')  #

    # model = Sequential()
    # model.add(LSTM(25, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(dropout_))
    # model.add(Conv1D(filters=3, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(dropout_))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(RepeatVector(n_outputs))
    # model.add(LSTM(15, activation='relu', return_sequences=True))
    # model.add(Dropout(dropout_))
    # model.add(TimeDistributed(Dense(1)))
    # optimizer = optimizers.Adam(lr=0.0001)
    # model.compile(optimizer=optimizer, loss='mse')

    # fit network                                                               #
    history = model.fit(X_train,  #
                        y_train,  #
                        epochs=epochs_,  #
                        batch_size=batch_size_,  #
                        # validation_data=(X_valid, y_valid),
                        validation_split=0.2,
                        verbose=2)  #
    #
    # Save model for later                                                      #
    model.save(folder_saving + model_saved)  #

    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.title("loss plots")
    plt.savefig(folder_saving+model_saved+"_loss_plots")
    plt.clf()

    return model



