import os
from SolarForecasting.ModulesLearning.ModulesCNN.Model import basic_CNN, DC_CNN_Model
from keras.optimizers import Adam, Nadam
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, EarlyStopping

# def train(X_train, y_train, folder_saving, model_saved,n_timesteps=1, n_features = 1, n_outputs=1):
#     batch_size = 2 ** 5
#     epochs = 20
#
#     X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
#     y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
#
#     print("train shape: ", X_train.shape)
#     print("train shape: ", y_train.shape)
#
#     model = build_model(n_outputs, n_features)
#     model.compile(Adam(), loss='mean_absolute_error')
#     print(model.summary())
#
#     history = model.fit(X_train,y_train,
#                         batch_size=batch_size,
#                         epochs=epochs,
#                         validation_split=0.2)
#     # save the model
#     model.save(folder_saving + model_saved)
#
#     plt.figure()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#
#     plt.xlabel('Epoch')
#     plt.ylabel('mean_absolute_error Loss')
#     plt.title('Loss Over Time')
#     plt.legend(['Train', 'Valid'])
#     plt.savefig(folder_saving + model_saved + "_loss_plots")
#     plt.clf()
#
#     return model


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
    y_train = y_train.reshape((y_train.shape[0],n_outputs, 1))

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

    verbose, epochs, batch_size = 0, 100, 16
    model.compile(loss='mse', optimizer='adam')

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


# def evaluate_DCNN(X_test, n_timesteps, n_features, predict_size, model):
#
#     X_test = X_test.reshape((X_test.shape[0],n_timesteps, n_features))
#     # pred_array = model.predict(X_test_initial) if predictions of training samples required
#
#     # forecast is created by predicting next future value based on previous predictions
#     # pred_array=[]
#     # for i in range(predict_size):
#     #     pred_array.append(model.predict(X_test[i]))
#     #
#     # return pred_array
#
#     for i in range(X_test.shape[0]):
#         X = X_test[i]
#         for j in range(predict_size):










