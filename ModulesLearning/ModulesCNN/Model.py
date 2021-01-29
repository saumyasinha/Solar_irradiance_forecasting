from keras.models import Model, load_model, Sequential
from keras.layers import Input, Lambda, MaxPooling1D
from keras.layers import Dense, Conv1D, Flatten, Dropout, Conv2D,Activation, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal
from keras.regularizers import l2

#
# def build_model(n_outputs, n_features):
#     n_filters = 128
#     filter_width = 5
#     dilation_rates = [2**i for i in range(4)]
#
#     # define an input history series and pass it through a stack of dilated causal convolutions
#     history_seq = Input(shape=(None, n_features))
#     x = history_seq
#
#     for dilation_rate in dilation_rates:
#         x = Conv1D(filters = n_filters,
#                    kernel_size=filter_width,
#                    padding='causal',
#                    dilation_rate=dilation_rate)(x)
#     # for Dense Layer:
#     # Input shape
#     # nD tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
#     #
#     # Output shape
#     # nD tensor with shape: (batch_size, ..., units).
#     # For instance, for a 2D input with shape  (batch_size, input_dim), the output would have shape (batch_size, units).
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(.5)(x)
#     x = Dense(16)(x)
#     x = Dense(1)(x)
#
#     # extract the last 14 time steps as the training target
#     def slice(x, seq_length):
#         return x[:, -seq_length:, :]
#
#     pred_seq_train = Lambda(slice, arguments={'seq_length': n_outputs})(x)
#
#     model = Model(history_seq, pred_seq_train)
#
#     return model


def basic_CNN(X_train):

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model = Sequential()

    ks1_first = 3
    ks1_second = 8

    ks2_first = 4
    ks2_second = 5

    model.add(Conv2D(filters=(3),
                     kernel_size=(ks1_first, ks1_second),
                     input_shape=input_shape,
                     padding='same',
                     kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.025))

    for _ in range(1):#2
        model.add(Conv2D(filters=(4),
                         kernel_size=(ks2_first, ks2_second),
                         padding='same',
                         kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.280))

    model.add(Flatten())

    for _ in range(2):#4
        model.add(Dense(64, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.435))

    # for _ in range(3):
    #     model.add(Dense(128, kernel_initializer='TruncatedNormal'))
    #     model.add(BatchNormalization())
    #     model.add(LeakyReLU())
    #     model.add(Dropout(0.372))

    model.add(Dense(128, kernel_initializer='TruncatedNormal'))#1024
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.793))

    model.add(Dense(1))

    return model


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual = input_

        layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                           dilation_rate=dilation,
                           activation='linear', padding='causal', use_bias=False,
                           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                              seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)

        layer_out = Activation('selu')(layer_out)

        skip_out = Conv1D(1, 1, activation='linear', use_bias=False,
                          kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                             seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_in = Conv1D(1, 1, activation='linear', use_bias=False,
                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                               seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)

        network_out = Add()([residual, network_in])

        return network_out, skip_out

    return f


def DC_CNN_Model(n_timesteps, n_features, n_outputs = 1):
    # input = Input(shape=(length, n_features))
    #
    # l1a, l1b = DC_CNN_Block(32, 2, 1, 0.001)(input)
    # l2a, l2b = DC_CNN_Block(32, 2, 2, 0.001)(l1a)
    # l3a, l3b = DC_CNN_Block(32, 2, 4, 0.001)(l2a)
    # l4a, l4b = DC_CNN_Block(32, 2, 8, 0.001)(l3a)
    # l5a, l5b = DC_CNN_Block(32, 2, 16, 0.001)(l4a)
    # # l6a, l6b = DC_CNN_Block(32, 2, 32, 0.001)(l5a)
    # # l6b = Dropout(0.8)(l6b)  # dropout used to limit influence of earlier data
    # # l7a, l7b = DC_CNN_Block(32, 2, 64, 0.001)(l6a)
    # # l7b = Dropout(0.8)(l7b)  # dropout used to limit influence of earlier data
    # l5b = Dropout(0.8)(l5b)
    #
    # l8 = Add()([l1b, l2b, l3b, l4b, l5b])#, l6b, l7b])
    #
    # l9 = Activation('relu')(l8)
    #
    # l21 = Conv1D(1, 1, activation='linear', use_bias=False,
    #              kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
    #              kernel_regularizer=l2(0.001))(l9)
    #
    # model = Model(input=input, output=l21)
    #
    # return model

    #Model 1
    n_filters = 32
    filter_width = 2
    dilation_rates = [2**i for i in range(5)]

    # define an input history series and pass it through a stack of dilated causal convolutions
    history_seq = Input(shape=(None, n_features))
    x = history_seq

    for dilation_rate in dilation_rates:
        x = Conv1D(filters = n_filters,
                   kernel_size=filter_width,
                   padding='causal',
                   dilation_rate=dilation_rate)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(1)(x)

    def slice(x, seq_length):

        return x[:, -seq_length:, :]

    pred_seq_train = Lambda(slice, arguments={'seq_length': n_outputs})(x)

    model = Model(history_seq, pred_seq_train)

    #model2:
    # convolutional layer oparameters
    # n_filters = 128
    # filter_width = 5
    # dilation_rates = [2 ** i for i in range(12)]
    #
    # # define an input history series and pass it through a stack of dilated causal convolutions
    # history_seq = Input(shape=(None, n_features))
    # x = history_seq
    #
    # for dilation_rate in dilation_rates:
    #     x = Conv1D(filters=n_filters,
    #                kernel_size=filter_width,
    #                padding='causal',
    #                dilation_rate=dilation_rate)(x)
    # # for Dense Layer:
    # # Input shape
    # # nD tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
    # #
    # # Output shape
    # # nD tensor with shape: (batch_size, ..., units).
    # # For instance, for a 2D input with shape  (batch_size, input_dim), the output would have shape (batch_size, units).
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(.8)(x)
    # x = Dense(64)(x)
    # x = Dense(1)(x)
    #
    # def slice(x, seq_length):
    #
    #     return x[:, -seq_length:, :]
    #
    # pred_seq_train = Lambda(slice, arguments={'seq_length': n_outputs})(x)
    #
    # model = Model(history_seq, pred_seq_train)

    return model
