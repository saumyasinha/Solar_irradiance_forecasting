# from keras.models import Model, load_model, Sequential
# from keras.layers import Input, Lambda, MaxPooling1D
# from keras.layers import Dense, Conv1D, Flatten, Dropout, Conv2D,Activation, Add
# from keras.layers.advanced_activations import LeakyReLU, ELU
# from keras.layers.normalization import BatchNormalization
# from keras.initializers import TruncatedNormal
# from keras.regularizers import l2

import torch
import torch.nn as nn
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, saving_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.saving_path = saving_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.saving_path)
        self.val_loss_min = val_loss

class Task_independent_module(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1)])


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

class ConvForecasterDilationLowRes(nn.Module):
    def __init__(self, input_dim, timesteps, folder_saving, model, quantile, alphas=None, outputs=None, valid=False):
        super(ConvForecasterDilationLowRes, self).__init__()
        self.quantile = quantile

        if self.quantile:
            assert outputs == len(alphas), "The outputs and the quantiles should be of the same dimension"

        self.input_dim = input_dim
        self.timesteps = timesteps
        self.alphas = alphas
        self.outputs = outputs
        self.valid = valid
        self.train_mode = False
        self.saving_path = folder_saving+model

        self.conv1 = nn.Conv1d(self.input_dim, 40, 2, stride=1)
        self.conv1_fn = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=1)

        # k × k filter is enlarged to k + (k − 1)(r − 1) with dilated stride r

        self.conv2 = nn.Conv1d(40, 80, 3, stride=1, dilation=2)
        self.conv2_fn = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(80, 128, 3, stride=1, dilation=4)
        self.conv3_fn = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=1)


        conv_layers = [ self.conv1,self.conv1_fn,self.avgpool1,self.conv2,self.conv2_fn,self.avgpool2,self.conv3,self.conv3_fn,self.avgpool3]
        # conv_layers = [self.conv1, self.conv1_fn, self.conv2, self.conv2_fn, self.conv3,
        #                self.conv3_fn]
        conv_module = nn.Sequential(*conv_layers)

        test_ipt = Variable(torch.zeros(1, self.input_dim, self.timesteps))
        test_out = conv_module(test_ipt)

        self.conv_output_size = test_out.size(1) * test_out.size(2)
        fc1_dim = self.conv_output_size+(self.input_dim*self.timesteps)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(fc1_dim, int(fc1_dim/2))#int(fc1_dim/4))
        self.fc1_fn = nn.Tanh()

        self.fc2 = nn.Linear(int(fc1_dim/2), int(fc1_dim/4))
        self.fc2_fn = nn.Tanh()


        # Attention Layer :
        self.conv_attn = nn.Conv1d(self.input_dim, 1, 1, stride=1)
        self.attn_layer = nn.Sequential(
            nn.Linear(self.conv_output_size+self.timesteps, self.timesteps*self.input_dim),
            nn.Tanh(),
            nn.Linear(self.timesteps*self.input_dim, self.timesteps),
            # nn.Linear(self.conv_output_size+self.timesteps, self.timesteps),
            nn.Softmax(dim=1)
        )


        self.fc3 = nn.Linear(int(fc1_dim/4), self.outputs)

        ## Adding task independent layers for multi-task learning
        # self.task_nets = nn.ModuleList()
        # for _ in range(self.outputs):
        #     self.task_nets.append(
        #         Task_independent_module(int(fc1_dim/2),int(fc1_dim/4))
        #          )


    def forward(self, xx):
        # print(xx.shape)
        output = self.conv1(xx)
        # print(output.shape)
        output = self.conv1_fn(output)
        output = self.avgpool1(output)
        # print(output.shape)

        output = self.conv2(output)
        # print(output.shape)
        output = self.conv2_fn(output)
        output = self.avgpool2(output)
        # print(output.shape)

        output = self.conv3(output)
        # print(output.shape)
        output = self.conv3_fn(output)
        output = self.avgpool3(output)
        # print(output.shape)
        output = output.reshape(-1, output.shape[1]*output.shape[2])
        # print("after convolution: ", output.shape)
        # Compute Context Vector
        xx_single = self.conv_attn(xx).reshape(-1, self.timesteps)
        # print("xx_single: ", xx_single.shape)
        attn_input = torch.cat((output, xx_single), dim=1)
        # print("attn_input: ", attn_input.shape)
        attention = self.attn_layer(attn_input).reshape(-1, 1, self.timesteps)

        # print("attention: ", attention.shape)
        x_attenuated = (xx * attention)
        # print("xx attentuated: ",x_attenuated.shape)
        x_attenuated = x_attenuated.reshape(-1, x_attenuated.shape[1]*x_attenuated.shape[2])


        output = torch.cat((output, x_attenuated), dim=1)
        # print("output concat with attenuated: ",output.shape)
        output = self.fc1(output)
        output = self.fc1_fn(output)
        if self.train_mode:
            output = self.dropout(output)

        output = self.fc2(output)
        output = self.fc2_fn(output)
        if self.train_mode:
            output = self.dropout(output)
        output = self.fc3(output)

        # output = torch.cat(
        #     tuple(task_model(output) for task_model in self.task_nets),
        #     dim=1
        # )

        return output

    def trainBatchwise(self, trainX, trainY, epochs, batch_size, lr=0.0001, validX=None,
                       validY=None, patience=None, verbose=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        samples = trainX.size()[0]
        losses = []
        valid_losses = []

        early_stopping = EarlyStopping(self.saving_path, patience=patience, verbose=True)

        for epoch in range(epochs):
            if self.train_mode is not True:
                self.train()
                self.train_mode = True

            indices = torch.randperm(samples)
            trainX, trainY = trainX[indices, :, :], trainY[indices]
            per_epoch_loss = 0
            count_train = 0
            for i in range(0, samples, batch_size):
                xx = trainX[i: i + batch_size, :, :]
                yy = trainY[i: i + batch_size]

                if torch.cuda.is_available():
                    xx, yy = xx.cuda(), yy.cuda()
                outputs = self.forward(xx)
                optimizer.zero_grad()
                if self.quantile:
                    loss = self.quantile_loss(outputs, yy)
                else:
                    loss = criterion(outputs, yy)

                ## train loss for multiple outputs or multi-task learning
                # total_loss = []
                # for n in range(self.outputs):
                #     y_pred = outputs[:, n]
                #     # calculate the batch loss
                #     loss = criterion(y_pred, yy[:, n])
                #     total_loss.append(loss)
                #
                # loss = sum(total_loss)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                per_epoch_loss+=loss.item()
                count_train+=1


            train_loss_this_epoch = per_epoch_loss/count_train
            losses.append(train_loss_this_epoch)

            if self.valid:
                self.train_mode = False
                self.eval()
                if torch.cuda.is_available():
                    validX,validY = validX.cuda(), validY.cuda()

                if self.quantile:
                    validYPred = self.forward(validX)
                    # validYPred = validYPred.cpu().detach().numpy()
                    # validYTrue = validY.cpu().detach().numpy()
                    valid_loss_this_epoch = self.quantile_loss(validYPred,validY).item()
                    # valid_loss = elf.crps_score(validYPred, validYTrue, np.arange(0.05, 1.0, 0.05))s
                    valid_losses.append(valid_loss_this_epoch)
                    print("Epoch: %d, loss: %1.5f and valid_loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))
                else:
                    validYPred = self.forward(validX)

                    # valid loss for multiple outputs or multi-task learning
                    # total_loss = []
                    # for n in range(self.outputs):
                    #     y_pred = validYPred[:, n]
                    #     # calculate the batch loss
                    #     validloss = criterion(y_pred, validY[:, n])
                    #     total_loss.append(validloss)
                    #
                    # validloss = sum(total_loss)
                    # valid_loss_this_epoch = validloss.item()
                    valid_loss_this_epoch = criterion(validYPred, validY).item()
                    # validYPred = validYPred.cpu().detach().numpy()
                    # validYTrue = validY.cpu().detach().numpy()
                    # valid_loss = np.sqrt(mean_squared_error(validYPred, validYTrue))
                    valid_losses.append(valid_loss_this_epoch)
                    print("Epoch: %d, train loss: %1.5f and valid loss : %1.5f" % (epoch, train_loss_this_epoch, valid_loss_this_epoch))

                # early_stopping(valid_loss, self)
                early_stopping(valid_loss_this_epoch, self)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                print("Epoch: %d, loss: %1.5f" % (epoch, train_loss_this_epoch))
        # load the last checkpoint with the best model
        # self.load_state_dict(torch.load('checkpoint.pt'))
        return losses, valid_losses

    def crps_score(self, outputs, target, alphas):
        loss = []
        for i, alpha in enumerate(alphas):
            output = outputs[:, i].reshape((-1, 1))
            covered_flag = (output <= target).astype(np.float32)
            uncovered_flag = (output > target).astype(np.float32)
            if i == 0:
                loss.append(np.mean(
                    ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)))
            else:
                loss.append(np.mean(
                    ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)))

        return 2*np.mean(np.array(loss))

    def quantile_loss(self, outputs, target):
        for i, alpha in zip(range(self.outputs), self.alphas):
            output = outputs[:, i].reshape((-1, 1))
            covered_flag = (output <= target).float()
            uncovered_flag = (output > target).float()
            if i == 0:
                loss = ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)
            else:
                loss += ((target - output) * alpha * covered_flag + (output - target) * (1 - alpha) * uncovered_flag)

        return torch.mean(loss)



## Keras models
# def basic_CNN(X_train):
#
#     ## 2D convolution (vanilla CNN)
#     input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
#
#     model = Sequential()
#
#     ks1_first = 3
#     ks1_second = 8
#
#     ks2_first = 4
#     ks2_second = 5
#
#     model.add(Conv2D(filters=(3),
#                      kernel_size=(ks1_first, ks1_second),
#                      input_shape=input_shape,
#                      padding='same',
#                      kernel_initializer='TruncatedNormal'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Dropout(0.025))
#
#     for _ in range(1):#2
#         model.add(Conv2D(filters=(4),
#                          kernel_size=(ks2_first, ks2_second),
#                          padding='same',
#                          kernel_initializer='TruncatedNormal'))
#         model.add(BatchNormalization())
#         model.add(LeakyReLU())
#         model.add(Dropout(0.280))
#
#     model.add(Flatten())
#
#     for _ in range(2):#4
#         model.add(Dense(64, kernel_initializer='TruncatedNormal'))
#         model.add(BatchNormalization())
#         model.add(LeakyReLU())
#         model.add(Dropout(0.435))
#
#     # for _ in range(3):
#     #     model.add(Dense(128, kernel_initializer='TruncatedNormal'))
#     #     model.add(BatchNormalization())
#     #     model.add(LeakyReLU())
#     #     model.add(Dropout(0.372))
#
#     model.add(Dense(128, kernel_initializer='TruncatedNormal'))#1024
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Dropout(0.793))
#
#     model.add(Dense(1))
#
#     return model
#
#
# def DC_CNN_Model(n_timesteps, n_features, n_outputs = 1):
#
#     #1D conv with dilation
#     n_filters = 32
#     filter_width = 2
#     dilation_rates = [2**i for i in range(8)] #changed from 5 in case of lag 1 hour
#
#     # define an input history series and pass it through a stack of dilated causal convolutions
#     history_seq = Input(shape=(None, n_features))
#     x = history_seq
#
#     for dilation_rate in dilation_rates:
#         x = Conv1D(filters = n_filters,
#                    kernel_size=filter_width,
#                    padding='causal',
#                    dilation_rate=dilation_rate, kernel_regularizer=l2(0.001))(x) ## add regularizer  kernel_regularizer=l2(0.001)
#
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(.4)(x) #0.2
#     x = Dense(1)(x)
#
#
#     def slice(x, seq_length):
#
#         return x[:, -seq_length:, :]
#
#     pred_seq_train = Lambda(slice, arguments={'seq_length': n_outputs})(x)
#
#     model = Model(history_seq, pred_seq_train)
#
#     #1D conv without dilation
#     # input_shape = (n_timesteps,n_features)
#     # model = Sequential()
#     # model.add(Conv1D(32, kernel_size=9,padding='same',
#     #                  activation='selu',
#     #                  input_shape=input_shape))
#     # model.add(MaxPooling1D(pool_size=2))
#     # model.add(Conv1D(64, kernel_size=7,padding='same',
#     #                  activation='selu')),
#     # model.add(MaxPooling1D(pool_size=2))
#     # model.add(Conv1D(128, kernel_size=5,padding='same',
#     #                  activation='selu')),
#     # model.add(MaxPooling1D(pool_size=2))
#     # model.add(Flatten())
#     # model.add(Dropout(0.3))#0.15
#     # # model.add(Dense(1000, activation='relu'))
#     # model.add(Dense(n_outputs, activation='relu'))
#
#     return model
#
#

