import torch
from ModulesLearning.ModuleLSTM.Model import MultiAttnHeadSimple
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
    quantile_forecaster = quantileLSTM(n_features, n_timesteps, folder_saving, model_saved, quantile, hidden_size=25,alphas = np.arange(0.05, 1.0, 0.05), outputs=19, valid=valid)
    if train_on_gpu:
        quantile_forecaster = quantile_forecaster.cuda()
        # point_foreaster = point_foreaster.cuda()

    print(quantile_forecaster)
    learning_rate = 1e-3

    epochs = 300
    batch_size = 32
    train_loss, valid_loss = quantile_forecaster.trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, patience=1000)
    loss_plots(train_loss,valid_loss,folder_saving,model_saved)


def test_LSTM(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):


    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
    X_test = torch.from_numpy(X_test).reshape(-1, n_timesteps, n_features)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_timesteps, n_features)



    quantile_forecaster = quantileLSTM(n_features, n_timesteps, folder_saving, model_saved, quantile,hidden_size=25,
                                                      alphas=np.arange(0.05, 1.0, 0.05), outputs=19, valid=True)

    quantile_forecaster.load_state_dict(torch.load(folder_saving + model_saved))

    quantile_forecaster.eval()

    y_pred = quantile_forecaster.forward(X_test)
    y_pred = y_pred.cpu().detach().numpy()
    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_forecaster.forward(X_valid)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0
    if quantile:
        test_crps = quantile_forecaster.crps_score(y_pred, y_test, np.arange(0.05, 1.0, 0.05))
        y_pred = y_pred[:,9]

        if X_valid is not None:
            valid_crps = quantile_forecaster.crps_score(y_valid_pred, y_valid, np.arange(0.05, 1.0, 0.05))
            y_valid_pred = y_valid_pred[:,9]

    return y_pred, y_valid_pred, valid_crps, test_crps


def train_transformer(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    # Size: [batch_size, seq_len, input_size]
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    # X_train = torch.from_numpy(X_train).reshape(-1, n_features, n_timesteps)
    X_train = torch.from_numpy(X_train).reshape(-1, n_timesteps, n_features)
    y_train = torch.from_numpy(y_train).reshape(-1,n_outputs)

    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        # X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_timesteps, n_features)
        y_valid = torch.from_numpy(y_valid).reshape(-1,n_outputs)

    print(X_train.shape, y_train.shape)

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    # alphas = np.arange(0.05, 1.0, 0.05)
    alphas =  np.arange(0.05,1,0.225)
    outputs = len(alphas)

    # point_foreaster = TransAm(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs, valid=valid)
    quantile_forecaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, alphas = alphas, outputs = outputs, valid=valid, output_seq_len = n_outputs)
    if train_on_gpu:
        quantile_forecaster = quantile_forecaster.cuda()
        # point_foreaster = point_foreaster.cuda()

    print(quantile_forecaster)
    learning_rate = 1e-5 #0.0001

    epochs = 250 #100
    batch_size = 8 #16 #32
    train_loss, valid_loss = quantile_forecaster.trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, n_outputs, patience=1000)

    loss_plots(train_loss,valid_loss,folder_saving,model_saved)


def test_transformer(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):


    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
    # X_test = torch.from_numpy(X_test).reshape(-1, n_features, n_timesteps)
    X_test = torch.from_numpy(X_test).reshape(-1, n_timesteps, n_features)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        # X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_timesteps, n_features)

    # point_foreaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs,
    #                           valid=True)

    # alphas = np.arange(0.05, 1.0, 0.05)
    alphas = np.arange(0.05, 1, 0.225)
    outputs = len(alphas)
    q50 = 2 #9

    quantile_forecaster = MultiAttnHeadSimple(n_features, n_timesteps, folder_saving, model_saved, quantile,
                                             alphas=alphas, outputs=outputs, valid=True, output_seq_len = n_outputs)

    quantile_forecaster.load_state_dict(torch.load(folder_saving + model_saved,map_location=torch.device('cpu')))

    quantile_forecaster.eval()

    y_pred = quantile_forecaster.forward(X_test)
    y_pred = y_pred.cpu().detach().numpy()
    y_valid_pred = None

    if X_valid is not None:
        y_valid_pred = quantile_forecaster.forward(X_valid)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()

    valid_crps, test_crps = 0.0, 0.0


    if quantile:
        if n_outputs>1:
            test_crps=[]
            for n in range(n_outputs):
                print(y_pred.shape)
                y_pred_n = y_pred[:, n, :]
                test_crps.append(quantile_forecaster.crps_score(y_pred_n, y_test[:,n], alphas))

            y_pred = y_pred[:,:,q50]

        else:
            test_crps=quantile_forecaster.crps_score(y_pred, y_test, alphas)
            y_pred = y_pred[:,q50]#changed from 9

        if X_valid is not None:

            if n_outputs > 1:
                valid_crps = []
                for n in range(n_outputs):
                    y_valid_pred_n = y_valid_pred[:, n, :]
                    valid_crps.append(quantile_forecaster.crps_score(y_valid_pred_n, y_valid[:, n], alphas))

                y_valid_pred = y_valid_pred[:, :, q50]

            else:
                valid_crps = quantile_forecaster.crps_score(y_valid_pred, y_valid, alphas)
                y_valid_pred = y_valid_pred[:, q50]

    return y_pred, y_valid_pred, valid_crps, test_crps
