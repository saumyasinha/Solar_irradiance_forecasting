import torch
from SolarForecasting.ModulesLearning.ModuleLSTM.Model import MultiAttnHeadSimple
import numpy as np
import matplotlib.pyplot as plt

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


def train_LSTM(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    # Size: [batch_size, seq_len, input_size]
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    X_train = torch.from_numpy(X_train).reshape(-1, n_timesteps, n_features)
    y_train = torch.from_numpy(y_train).reshape(-1, n_outputs)

    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_timesteps, n_features)
        y_valid = torch.from_numpy(y_valid).reshape(-1, n_outputs)

    print(X_train.shape, y_train.shape)

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    # point_foreaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs, valid=valid)
    quantile_foreaster = quantileLSTM(n_features, n_timesteps, folder_saving, model_saved, quantile, hidden_size=25,alphas = np.arange(0.05, 1.0, 0.05), outputs=19, valid=valid)
    if train_on_gpu:
        quantile_foreaster = quantile_foreaster.cuda()
        # point_foreaster = point_foreaster.cuda()

    print(quantile_foreaster)
    learning_rate = 1e-3

    epochs = 300
    batch_size = 32
    train_loss, valid_loss = quantile_foreaster.trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, patience=1000)
    loss_plots(train_loss,valid_loss,folder_saving,model_saved)


def test_LSTM(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):


    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
    X_test = torch.from_numpy(X_test).reshape(-1, n_timesteps, n_features)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_timesteps, n_features)



    quantile_foreaster = quantileLSTM(n_features, n_timesteps, folder_saving, model_saved, quantile,hidden_size=25,
                                                      alphas=np.arange(0.05, 1.0, 0.05), outputs=19, valid=True)

    quantile_foreaster.load_state_dict(torch.load(folder_saving + model_saved))

    quantile_foreaster.eval()

    y_pred = quantile_foreaster.forward(X_test)
    y_pred = y_pred.cpu().detach().numpy()
    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_foreaster.forward(X_valid)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0
    if quantile:
        test_crps = quantile_foreaster.crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))
        y_pred = y_pred[:,9]

        if X_valid is not None:
            valid_crps = quantile_foreaster.crps_score(y_valid_pred, y_valid, np.arange(0.05, 1.0, 0.05))
            y_valid_pred = y_valid_pred[:,9]

    return y_pred, y_valid_pred, valid_crps, test_crps


def train_transformer(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    # Size: [batch_size, seq_len, input_size]
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    # X_train = torch.from_numpy(X_train).reshape(n_timesteps, -1, n_features)
    X_train = torch.from_numpy(X_train).reshape(-1, n_features, n_timesteps)
    y_train = torch.from_numpy(y_train).reshape(-1,n_outputs)

    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        # X_valid = torch.from_numpy(X_valid).reshape(n_timesteps, -1, n_features)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)
        y_valid = torch.from_numpy(y_valid).reshape(-1,n_outputs)

    print(X_train.shape, y_train.shape)

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    # point_foreaster = TransAm(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs, valid=valid)
    quantile_foreaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, alphas = np.arange(0.05, 1.0, 0.05), outputs=19, valid=valid)
    if train_on_gpu:
        quantile_foreaster = quantile_foreaster.cuda()
        # point_foreaster = point_foreaster.cuda()

    print(quantile_foreaster)
    learning_rate = 0.0001

    epochs = 100
    batch_size = 16
    train_loss, valid_loss = quantile_foreaster.trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, patience=1000)
    loss_plots(train_loss,valid_loss,folder_saving,model_saved)


def test_transformer(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):


    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
    # X_test = torch.from_numpy(X_test).reshape(n_timesteps, -1, n_features)
    X_test = torch.from_numpy(X_test).reshape(-1, n_features, n_timesteps)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        # X_valid = torch.from_numpy(X_valid).reshape(n_timesteps, -1, n_features)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)

    # point_foreaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs,
    #                           valid=True)

    quantile_foreaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile,
                                             alphas=np.arange(0.05, 1.0, 0.05), outputs=19, valid=True)

    quantile_foreaster.load_state_dict(torch.load(folder_saving + model_saved))

    quantile_foreaster.eval()

    y_pred = quantile_foreaster.forward(X_test)
    y_pred = y_pred.cpu().detach().numpy()
    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_foreaster.forward(X_valid)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0
    if quantile:
        test_crps = quantile_foreaster.crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))
        y_pred = y_pred[:,9]

        if X_valid is not None:
            valid_crps = quantile_foreaster.crps_score(y_valid_pred, y_valid, np.arange(0.05, 1.0, 0.05))
            y_valid_pred = y_valid_pred[:,9]

    return y_pred, y_valid_pred, valid_crps, test_crps
