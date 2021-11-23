import os
# from SolarForecasting.ModulesLearning.ModulesCNN.Model import basic_CNN, DC_CNN_Model
import torch
from ModulesLearning.ModulesCNN.Model import ConvForecasterDilationLowRes,trainBatchwise, crps_score
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
#from torchvision import transforms


def basic_CNN_train(X_train, y_train, folder_saving, model_saved):

    # if os.path.isfile(folder_saving+model_saved) == True:
    #     model = load_model(folder_saving+model_saved)
    #     return model

    epochs = 100
    bs = 64
    lr = 1e-3  # 6e-4

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    model = basic_CNN(X_train)
    print(model.summary())


    nadam = Nadam(lr=lr)

    # compile & fit
    model.compile(optimizer=nadam, loss=['mse'])

    early_stopping_monitor = EarlyStopping(patience=5000)

    # checkpoint
    filepath = folder_saving+model_saved
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # epoch_size = 56
    # schedule = SGDRScheduler(min_lr=9e-7,
    #                          max_lr=4.3e-3,
    #                          steps_per_epoch=np.ceil(epoch_size / bs),
    #                          lr_decay=0.9,
    #                          cycle_length=5,  # 5
    #                          mult_factor=1.5)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.2,
              verbose=1, callbacks=[early_stopping_monitor, checkpoint])

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('mean_swaured_error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train', 'Valid'])
    plt.savefig(folder_saving + model_saved + "_loss_plots")
    plt.clf()

    return model

def train_DCNN(X_train,y_train, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):
    # timeseries input is 1-D numpy array
    # forecast_size is the forecast horizon
    #
    # if os.path.isfile(folder_saving+model_saved) == True:
    #     model = load_model(folder_saving+model_saved)
    #     return model

    X_train = X_train.reshape((X_train.shape[0],n_timesteps, n_features))
    # y_train = y_train.reshape((y_train.shape[0],n_outputs, 1))

    print(X_train.shape, y_train.shape)
    model = DC_CNN_Model(n_timesteps, n_features, n_outputs)
    print('\n\nModel with input size {}, output size {}'.
          format(model.input_shape, model.output_shape))

    print(model.summary())

    # adam = Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None,
    #             decay=0.0, amsgrad=False)
    #
    # model.compile(loss='mae', optimizer=adam)
    #
    # history = model.fit(X_train, y_train, epochs=3000)
    #
    # adam = Adam(lr=0.00075)

    verbose, epochs, batch_size = 0, 100, 32
    # model.compile(loss='mse', optimizer='adam')
    model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd')

    filepath = folder_saving + model_saved
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                        verbose=verbose, callbacks=[checkpoint])

    # model.save(folder_saving + model_saved)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('mean_squared_error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train', 'Valid'])
    plt.savefig(folder_saving + model_saved + "_loss_plots")
    plt.clf()

    return model


def loss_plots(train_loss, valid_loss, folder_saving, loss_type=""):
    epochs = range(1, len(train_loss)+1)
    # train_loss = train_loss[1:]
    # valid_loss = valid_loss[1:]
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training', 'validation'], loc='lower right')

    plt.savefig(folder_saving+"loss_plots_"+loss_type)
    plt.close()

def transform(X):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    X[:, 0, :, :] = (X[:, 0, :, :] - mean[0]) / std[0]
    X[:, 1, :, :] = (X[:, 1, :, :] - mean[1]) / std[1]
    X[:, 2, :, :] = (X[:, 2, :, :] - mean[2]) / std[2]

    return X

def train_DCNN_with_attention(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    X_train = torch.from_numpy(X_train) #.reshape(-1, n_features, n_timesteps)
    y_train = torch.from_numpy(y_train).reshape(-1, n_outputs)

    # X_train.unsqueeze_(1)
    # X_train = X_train.repeat(1, 3, 1, 1)
    #
    # X_train = transform(X_train)
    #
    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid) #.reshape(-1, n_features, n_timesteps)
        y_valid = torch.from_numpy(y_valid).reshape(-1, n_outputs)
        # X_valid.unsqueeze_(1)
        # X_valid = X_valid.repeat(1, 3, 1, 1)
        # X_valid = transform(X_valid)


    print(X_train.shape, y_train.shape)

    # point_forecaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs, valid=valid)
    learning_rate =1e-4 #1e-5 # 1e-4#1e-5#changed from 1e-5

    epochs = 300 #350 #400 #300#400 
    batch_size = 16#4  #32


    train_loss, valid_loss = trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, n_outputs,n_features, n_timesteps, folder_saving, model_saved, quantile, alphas = np.arange(0.05, 1.0, 0.05), outputs=19, valid=valid, patience=1000)
    loss_plots(train_loss,valid_loss,folder_saving,model_saved)





def test_DCNN_with_attention(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, folder_saving, model_saved, X_before_normalized=None, lead=None, n_outputs = 1):

    if X_test is not None:
        X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
        X_test = torch.from_numpy(X_test) #.reshape(-1, n_features, n_timesteps)

        # X_test.unsqueeze_(1)
        # X_test = X_test.repeat(1, 3, 1, 1)
        #
        # X_test = transform(X_test)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid) #.reshape(-1, n_features, n_timesteps)

        # X_valid.unsqueeze_(1)
        # X_valid = X_valid.repeat(1, 3, 1, 1)
        # X_valid = transform(X_valid)

        # point_forecaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile,
        #                                                outputs=n_outputs, valid=True)

    quantile_forecaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile,
                                                       alphas=np.arange(0.05, 1.0, 0.05), outputs=19, valid=True)

    # alphas = np.arange(0.05, 1.0, 0.05)
    # quantile_forecaster = Custom_resnet(n_features, n_timesteps, outputs=len(alphas))
    quantile_forecaster.load_state_dict(torch.load(folder_saving + model_saved, map_location=torch.device('cpu')))

    quantile_forecaster.eval()

    # if torch.cuda.is_available():
    #   X_test, X_valid = X_test.cuda(),X_valid.cuda()
    #  quantile_forecaster = quantile_forecaster.cuda()

    y_pred = None
    if X_test is not None:
        y_pred = quantile_forecaster.forward(X_test, n_outputs)
        y_pred = y_pred.cpu().detach().numpy()
    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_forecaster.forward(X_valid, n_outputs)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0

    if quantile:
        if X_test is not None:
            if X_before_normalized is not None:
                clearsky = y_test[:,1].reshape(y_test.shape[0],1)
                true = y_test[:0].reshape(y_test.shape[0],1) #np.roll(y_test, lead)
                y_test = np.multiply(true, clearsky)
                pred = y_pred #np.roll(y_pred, lead,axis=0)
                y_pred = np.multiply(pred, clearsky)

                test_crps=crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))
                y_pred = y_pred[:,9]#changed from 9
            else:
                y_test = y_test[:0].reshape(y_test.shape[0], 1)
                test_crps = crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))

        if X_valid is not None:
            y_valid = y_valid[:0].reshape(y_valid.shape[0], 1)
            valid_crps = crps_score(y_valid_pred, y_valid, np.arange(0.05, 1.0, 0.05))
            y_valid_pred = y_valid_pred[:, 9]  # changed from 9


    # print(valid_crps)
    return y_pred, y_valid_pred, valid_crps, test_crps, y_test










