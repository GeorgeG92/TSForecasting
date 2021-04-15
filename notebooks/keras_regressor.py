from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf

def compileModel(epochs, L2, alpha, batch_size, dropout, 
	learning_rate, optimizer, init='glorot_uniform'):
        input_shape = (2233, 504, 4) 
        model = Sequential()                           # LSTM input layer MUST be 3D - (samples, timesteps, features)
        model.add(LSTM(504,
                #return_sequences=True,          # necessary for stacked LSTM layers
                input_shape=(input_shape[1], input_shape[2])))
        #model.add(LSTM(10))
        model.add(Dropout(dropout))
        model.add(Dense(256, 
                kernel_initializer = init,
                kernel_regularizer=l2(L2), 
                activation=LeakyReLU(alpha=alpha)))
        model.add(Dropout(dropout))
        model.add(Dense(168))

        model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
        return model