import torch
from ModulesLearning.ModuleLSTM.Model import TransAm,trainBatchwise,crps_score, quantileLSTM
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def loss_plots(train_loss, valid_loss, folder_saving, loss_type=""):
    epochs = range(2, len(train_loss)+1)
    train_loss = train_loss[1:]
    valid_loss = valid_loss[1:]
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training', 'validation'], loc='lower right')

    plt.savefig(folder_saving+"loss_plots_"+loss_type)
    plt.close()


def train_LSTM(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, hidden_size, batch_size, epochs, lr, alphas, q50, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    # Size: [batch_size, seq_len, input_size]
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    X_train = torch.from_numpy(X_train)  # .reshape(-1, n_timesteps, n_features)
    X_train = X_train.transpose(1, 2)
    y_train = torch.from_numpy(y_train).reshape(-1, n_outputs)

    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid)#.reshape(-1, n_timesteps, n_features)
        X_valid = X_valid.transpose(1, 2)
        y_valid = torch.from_numpy(y_valid).reshape(-1, n_outputs)

    print(X_train.shape, y_train.shape)

    outputs = len(alphas)

    learning_rate = lr

    epochs = epochs
    batch_size = batch_size

    train_loss, valid_loss = trainBatchwise(X_train, y_train, epochs, batch_size, learning_rate, X_valid, y_valid,
                                            n_outputs, n_features, n_timesteps, folder_saving, model_saved, quantile,
                                            hidden_size, alphas=alphas, outputs=outputs, valid=valid,
                                            patience=1000)

    loss_plots(train_loss, valid_loss, folder_saving, model_saved)



def test_LSTM(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, hidden_size, alphas, q50, folder_saving, model_saved, X_before_normalized=None, lead=None, n_outputs = 1):

    if X_test is not None:
        X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
        X_test = torch.from_numpy(X_test) #.reshape(-1, n_timesteps, n_features)
        X_test = X_test.transpose(1, 2)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid) #.reshape(-1, n_timesteps, n_features)
        X_valid = X_valid.transpose(1, 2)

    outputs = len(alphas)

    quantile_forecaster = quantileLSTM(n_features, n_timesteps, folder_saving, model_saved, quantile, hidden_size, alphas = alphas, outputs = outputs, valid=True)

    quantile_forecaster.load_state_dict(torch.load(folder_saving + model_saved, map_location=torch.device('cpu')))

    quantile_forecaster.eval()

    y_pred = None
    if X_test is not None:
        y_pred = quantile_forecaster.forward(X_test)
        y_pred = y_pred.cpu().detach().numpy()

    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_forecaster.forward(X_valid)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0
    if quantile:
        if X_test is not None:
            if X_before_normalized is not None:
                clearsky = y_test[:, 1].reshape(y_test.shape[0], 1)
                true = y_test[:,0].reshape(y_test.shape[0], 1)  # np.roll(y_test, lead)
                y_test = np.multiply(true, clearsky)
                pred = y_pred  # np.roll(y_pred, lead,axis=0)
                y_pred = np.multiply(pred, clearsky)

                test_crps = crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))
                y_pred = y_pred[:, 9]  # changed from 9
            else:
                y_test = y_test[:,0].reshape(y_test.shape[0], 1)
                test_crps = crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))

        if X_valid is not None:
            y_valid = y_valid[:,0].reshape(y_valid.shape[0], 1)
            valid_crps = crps_score(y_valid_pred, y_valid, np.arange(0.05, 1.0, 0.05))
            y_valid_pred = y_valid_pred[:, 9]  # changed from 9


    return y_pred, y_valid_pred, valid_crps, test_crps


def train_transformer(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, n_layers, num_heads, d_model, batch_size, epochs, lr, alphas, q50, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    # Size: [batch_size, seq_len, input_size]
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    # X_train = torch.from_numpy(X_train).reshape(-1, n_features, n_timesteps)
    X_train = torch.from_numpy(X_train) #.reshape(-1, n_timesteps, n_features)
    X_train = X_train.transpose(1,2)
    y_train = torch.from_numpy(y_train).reshape(-1,n_outputs)

    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        # X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)
        X_valid = torch.from_numpy(X_valid) #.reshape(-1, n_timesteps, n_features)
        X_valid = X_valid.transpose(1, 2)
        y_valid = torch.from_numpy(y_valid).reshape(-1,n_outputs)

    print(X_train.shape, y_train.shape)

    # train_on_gpu = torch.cuda.is_available()
    # print(train_on_gpu)
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    outputs = len(alphas)
    #
    # # point_foreaster = TransAm(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs, valid=valid)
    # quantile_forecaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, n_layers, factor, alphas = alphas, outputs = outputs, valid=valid, output_seq_len = n_outputs, num_heads=num_heads, d_model=d_model)
    # if train_on_gpu:
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         quantile_forecaster = nn.DataParallel(quantile_forecaster)
    #
    #     quantile_forecaster = quantile_forecaster.cuda()
    #
    #     # point_foreaster = point_foreaster.cuda()
    #
    # print(quantile_forecaster)
    learning_rate = lr

    epochs = epochs
    batch_size = batch_size

    train_loss, valid_loss = trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, n_outputs,n_features, n_timesteps, folder_saving, model_saved, quantile, n_layers,  alphas = alphas, outputs = outputs, valid=valid, output_seq_len = n_outputs, num_heads=num_heads, d_model=d_model, patience=1000)

    loss_plots(train_loss,valid_loss,folder_saving,model_saved)


def test_transformer(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, n_layers,  num_heads, d_model, alphas, q50, folder_saving, model_saved, X_before_normalized=None, lead=None, n_outputs = 1):

    if X_test is not None:
        X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
        X_test = torch.from_numpy(X_test) #.reshape(-1, n_timesteps, n_features)
        X_test = X_test.transpose(1, 2)


    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        # X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)
        X_valid = torch.from_numpy(X_valid) #.reshape(-1, n_timesteps, n_features)
        X_valid = X_valid.transpose(1, 2)


    # point_foreaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs,
    #                           valid=True)

    outputs = len(alphas)


    # quantile_forecaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile,n_layers,factor,
    #                                          alphas=alphas, outputs=outputs, valid=True, output_seq_len = n_outputs, num_heads=num_heads, d_model=d_model)
    #
    #                                                    # changed np.arange step size from 0.05 to 0.1

    quantile_forecaster = TransAm(n_features, n_timesteps, folder_saving, model_saved, quantile, alphas=alphas,
                                  outputs=outputs, valid=True, num_heads=num_heads, d_model=d_model,
                                  num_layers=n_layers)

    quantile_forecaster.load_state_dict(torch.load(folder_saving + model_saved,map_location=torch.device('cpu')))

    quantile_forecaster.eval()

    y_pred = None
    if X_test is not None:
        y_pred = quantile_forecaster.forward(X_test)
        y_pred = y_pred.cpu().detach().numpy()

    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_forecaster.forward(X_valid)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0
    if quantile:
        if X_test is not None:
            if X_before_normalized is not None:
                clearsky = y_test[:, 1].reshape(y_test.shape[0], 1)
                true = y_test[:,0].reshape(y_test.shape[0], 1)  # np.roll(y_test, lead)
                y_test = np.multiply(true, clearsky)
                pred = y_pred  # np.roll(y_pred, lead,axis=0)
                y_pred = np.multiply(pred, clearsky)

                test_crps = crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))
                y_pred = y_pred[:, 9]  # changed from 9
            else:
                y_test = y_test[:,0].reshape(y_test.shape[0], 1)
                test_crps = crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))

        if X_valid is not None:
            y_valid = y_valid[:,0].reshape(y_valid.shape[0], 1)
            valid_crps = crps_score(y_valid_pred, y_valid, np.arange(0.05, 1.0, 0.05))
            y_valid_pred = y_valid_pred[:, 9]  # changed from 9

    return y_pred, y_valid_pred, valid_crps, test_crps








