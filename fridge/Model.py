from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv1D, Bidirectional, TimeDistributed, Flatten, GRU
from keras.utils import plot_model
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2


def create_model(kernal_size, time_stamp, model_name, batch_size=None, ):
    '''Creates the RNN module
    '''
    if model_name == "Seq2Seq":

        model = Sequential()

        # 1D Conv
        model.add(
            Conv1D(16, kernal_size, activation="relu", batch_input_shape=(batch_size, time_stamp, 2), padding="same",
                   strides=1))

        # Bi-directional LSTMs
        model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Dropout(rate=0.5))
        model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Dropout(rate=0.5))
        # Fully Connected Layers
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(Dropout(rate=0.5))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

        #opt = Adam(lr=0.0001)
        model.compile(loss='mse', optimizer="adam")
        plot_model(model, to_file='model_{}.png'.format(model_name), show_shapes=True)

        return model

    if model_name == "Seq2P_GRU":
        '''Creates the RNN module described in the paper
        '''
        model = Sequential()

        # 1D Conv
        model.add(
            Conv1D(16, kernal_size, activation="relu", batch_input_shape=(batch_size, time_stamp, 2), padding="same",
                   strides=1))

        # Bi-directional GRU
        model.add(Bidirectional(GRU(64, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Dropout(rate=0.5))
        model.add(Bidirectional(GRU(128, return_sequences=False, stateful=False), merge_mode='concat'))
        model.add(Dropout(rate=0.5))

        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='linear'))

        #opt = Adam(lr=0.0001)
        model.compile(loss='mse', optimizer="adam")
        plot_model(model, to_file='model_{}.png'.format(model_name), show_shapes=True)

        return model

    if model_name == "CONV":
        '''Creates and returns the ShortSeq2Point Network
        Based on: https://arxiv.org/pdf/1612.09106v3.pdf
        '''
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(30, 10, activation='relu', input_shape=(time_stamp, 2), padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
        model.add(Dropout(0.5))
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file='model_{}.png'.format(model_name), show_shapes=True)

        return model





