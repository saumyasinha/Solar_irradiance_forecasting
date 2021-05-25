import os
# from SolarForecasting.ModulesLearning.ModulesCNN.Model import basic_CNN, DC_CNN_Model
import torch
from ModulesLearning.ModulesCNN.Model import ConvForecasterDilationLowRes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




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


def train_DCNN_with_attention(quantile, X_train, y_train, X_valid, y_valid, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):

    valid = True

    # if valid:
    #     X_train, X_valid, y_train, y_valid = train_test_split(
    #         X_train, y_train, test_size=0.15, random_state=42)

    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
    X_train = torch.from_numpy(X_train).reshape(-1, n_features, n_timesteps)
    y_train = torch.from_numpy(y_train).reshape(-1, n_outputs)

    if valid:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)
        y_valid = torch.from_numpy(y_valid).reshape(-1, n_outputs)

    print(X_train.shape, y_train.shape)

    train_on_gpu = torch.cuda.is_available()
    print(train_on_gpu)

    # point_foreaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile, outputs=n_outputs, valid=valid)
    quantile_foreaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile, alphas = np.arange(0.05, 1.0, 0.05), outputs=19, valid=valid)
    if train_on_gpu:
        quantile_foreaster = quantile_foreaster.cuda()
        # point_foreaster = point_foreaster.cuda()

    print(quantile_foreaster)
    learning_rate = 1e-6#0.0001 orig

    epochs = 300 #200 for orig
    batch_size = 32
    train_loss, valid_loss = quantile_foreaster.trainBatchwise(X_train, y_train, epochs, batch_size,learning_rate, X_valid, y_valid, patience=1000)
    loss_plots(train_loss,valid_loss,folder_saving,model_saved)


def test_DCNN_with_attention(quantile, X_valid, y_valid, X_test, y_test, n_timesteps, n_features, folder_saving, model_saved, n_outputs = 1):


    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
    X_test = torch.from_numpy(X_test).reshape(-1, n_features, n_timesteps)

    if X_valid is not None:
        X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.float32)
        X_valid = torch.from_numpy(X_valid).reshape(-1, n_features, n_timesteps)



    # point_foreaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile,
    #                                                outputs=n_outputs, valid=True)

    quantile_foreaster = ConvForecasterDilationLowRes(n_features, n_timesteps, folder_saving, model_saved, quantile,
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












