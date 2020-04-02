from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
import warnings


class LSTMModel():
    def __init__(self, df, train, config, exportpath='../output/LSTM'):
        self.train = train
        self.testsize = config.testSize
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        self.exportpath = exportpath
        self.plotTimeSeries(df)
        trainX, trainY, testX, testY = self.preProcess(df)
        self.buildModel(config, trainX, trainY, testX, testY)
        self.evaluatePlot(df, trainX, trainY, testX, testY)


    def plotTimeSeries(self, df):
        train_size = int(len(df) * (1-self.testsize))
        train, test = df[0:train_size], df[train_size:len(df)]
        plt.figure(figsize=(14, 14), dpi=200)
        plt.plot(train['requests'])
        plt.plot(test['requests'])
        plt.xticks(rotation=45)
        plt.savefig(self.exportpath+"/initialplots.png")
        plt.close()

    def preProcess(self, df):
        #print("Preprocess:")
        computeCols = [col for col in df.columns if col!='request_date']
        df[computeCols] = df[computeCols].astype('float32')
        # 0 - remove TS index
        df2 = df.reset_index()
        if 'request_date' in df2.columns:
            df2 = df2.drop(columns=['request_date'])
        if 'index' in df2.columns:
            df2 = df2.drop(columns=['index'])

        # 1 scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        df2 = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)
        self.scaler = scaler
        #df2.head()

        # 2 formulate as supervised ML - necessary for multivariate forecasting
        df2['label'] = df2['requests'].shift(-1)                 # add next month's deposits as label!      #fix!
        df2.dropna(inplace=True)                                 # last row has a null label! -> drop it    #fix!

        # 3 Train/set split
        test_size = self.testsize*len(df2)
        train_size = len(df2) - test_size
        labels = df2['label']
        df2.drop(columns=['label'], inplace=True)
        trainX, testX = df2.loc[0:train_size, :], df2.loc[train_size:len(df2),:]          # spit the features
        trainY, testY = labels.loc[0:train_size], labels.loc[train_size:len(labels)]      # split the labels

        # 4 Reshape to expected LSTM input shape
        trainX_array = np.array(trainX)
        trainY_array = np.array(trainY)
        trainX_array = trainX_array.reshape(trainX.shape[0], 1, trainX.shape[1])
        input_shape = trainX_array.shape
        self.inputshape = input_shape

        testX_array = np.array(testX)
        testY_array = np.array(testY)
        testX_array = testX_array.reshape(testX.shape[0], 1, testX.shape[1])
        return trainX_array, trainY_array, testX_array, testY_array



    def defineModel(self, config):
        # LSTM model 1
        input_shape = self.inputshape
        model = Sequential()                           # LSTM input layer MUST be 3D - (samples, timesteps, features)
        model.add(LSTM(50, activation='relu',
                       return_sequences=True,          # necessary for stacked LSTM layers
                       input_shape=(input_shape[1], input_shape[2])))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model


    def trainModel(self, config, trainX, trainY, testX, testY ):           # manually found best parameters
        print("Training model on TS data...")
        batch_size = 16
        epochs = 20
        model = self.model
        history = model.fit(trainX, trainY,
                                validation_data=(testX, self.testY),
                                epochs=epochs, batch_size=batch_size,
                                verbose=1)
        # enforce model to disk (.h5)
        model.save_weights('../model/lstm.h5')
        self.history = history
        self.plotTrainHistory()
        self.model = model

    def loadModel(self, path):
        # load from disk
        print("Loading model weights from Disk...")
        self.model.load_weights(path)

    def buildModel(self, config, trainX, trainY, testX, testY ):
        print("Building model...")
        self.defineModel(config)
        if self.train:             # load
            self.trainModel(config, trainX, trainY, testX, testY)
        else:
            if os.path.exists(self.exportpath+'/lstm.h5'):
                self.loadModel(self.exportpath+'/lstm.h5')
            else:
                # log
                print("No saved model detected, forcing Training")
                self.trainModel()


    def plotTrainHistory(self):
        print("Exporting Train History...")
        history = self.history
        plt.figure(figsize=(9, 7), dpi=200)
        plt.plot(history.history["loss"], 'darkred', label="Train")
        plt.plot(history.history["val_loss"], 'darkblue', label="Validation")
        #plt.plot(history.history["val_loss"], 'darkblue', label="Test")
        plt.title("Loss over epoch")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.exportpath+'./LeaningCurves.png')
        plt.close()


    def evaluatePlot(self, df, trainX, trainY, testX, testY):
        model = self.model
        trainX_array = trainX
        testX_array = testX
        scaler = self.scaler

        trainPredict = np.array(model.predict(trainX_array))         # forecast based on training data
        testPredict = np.array(model.predict(testX_array))           # forecast of test (unseen data)

        testX_array = testX_array.reshape((testX_array.shape[0], testX_array.shape[2]))

        # invert scale for train forecast (train)
        trainX_array = trainX_array.reshape((trainX_array.shape[0], trainX_array.shape[2]))
        trainPredict = trainPredict.reshape((len(trainPredict), 1))
        inv_x = np.concatenate((trainPredict, trainX_array[:, 1:]), axis=1)
        inv_x = scaler.inverse_transform(inv_x)
        inv_x = inv_x[:,0]

        # invert scaling for forecast (test)
        inv_yhat = np.concatenate((testPredict, testX_array[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        # invert scaling for actual values
        testY_array = testY
        testY_array = testY_array.reshape((len(testY_array), 1))
        inv_y = np.concatenate((testY_array, testX_array[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]


        # calculate RMSE based on original-scaled data
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('\tPerformance (RMSE) on Test Data is : %.3f' % rmse)


        ### Merge inv_y with trainY_pred!
        wholeforecast = np.concatenate((inv_x, inv_yhat), axis=0)
        len(wholeforecast)

        # Plot against original data
        fig, ax1 = plt.subplots(figsize=(14,11))
        ax1.plot(np.array(df['requests']), label='Actual Requests')
        ax1.plot(wholeforecast, label = 'Forecasted Requests')
        plt.axvline(x=(1-self.testsize)*len(df), label='train/test split', c='k')
        ax1.set_title('Requests vs Time')
        ax1.set_ylabel('Requests')
        ax1.set_xlabel('hours')
        L=ax1.legend() #create and get the legend
        plt.savefig(self.exportpath+'/actualVSpredicted.jpg')
        plt.close()
