from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
from keras.regularizers import l2
from keras.metrics import mse
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import math
import yaml
import warnings


class LSTMModel():
    def __init__(self, df, train, config, exportpath=os.path.join('..', 'output', 'LSTM')):
        self.train = train
        self.verbose = config['Forecasting']['verbose']
        if 'Optimal LSTM Parameters' in config:
            self.optimExists = True
            self.bestParams = config['Optimal LSTM Parameters']
        else:
            self.optimExists = False
        self.exploreParams = config['Forecasting']['LSTM']['GridSearchCV']
        self.stepsIn = config['Forecasting']['stepsIn']
        self.stepsOut = config['Forecasting']['stepsOut']
        self.gscvDict = config['Forecasting']['LSTM']
        self.testsize = (self.stepsIn+self.stepsOut)
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        self.exportpath = exportpath
        print("LSTM Modeling")
        #self.plotTimeSeries(df)
        #print(df.isnull().values.any())
        train_X, train_Y, test_X, test_Y = self.preProcess(df)
        model = self.buildModel(train_X, train_Y)
        self.evaluatePlot(model, df, train_X, train_Y, test_X, test_Y)


    def plotTimeSeries(self, df):
        train_size = int(len(df)-self.testSize)
        train, test = df[0:train_size], df[train_size:len(df)]
        plt.figure(figsize=(14, 14), dpi=200)
        plt.plot(train['requests'])
        plt.plot(test['requests'])
        plt.xticks(rotation=45)
        plt.savefig(os.path.exists(self.exportpath, "initialplots.png"))
        plt.close()

    def preProcess(self, df):
        def split_sequences(sequences, n_steps_in, n_steps_out):
            X, y = list(), list()
            for i in range(len(sequences)):
                end_ix = i + n_steps_in
                out_end_ix = end_ix + n_steps_out
                if out_end_ix > len(sequences):
                    break
                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 0]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

        stepsIn = self.stepsIn
        stepsOut = self.stepsOut
        testSize = self.testsize

        computeCols = [col for col in df.columns if col!='request_date']
        df[computeCols] = df[computeCols].astype('float32')
        df2 = df[['requests', 'Temperature', 'Precipitation', 'WindSpeed']]

        # Scale & Formulate as a Supervised  Learning method
        scalers = []
        for col in df2.columns:
            scaler = StandardScaler()
            df2[col] = pd.DataFrame(scaler.fit_transform(np.array(df2[col]).reshape(-1, 1)))
            scalers.append(scaler)

        self.scaler = scalers[0]
        data = np.array(df2)
        X, Y = split_sequences(data, stepsIn, stepsOut)

        # Train/set split
        trainsize = len(df2)-testSize
        train_X = X[:trainsize, :]
        train_Y = Y[:trainsize]
        test_X = X[trainsize:, :]
        test_Y = Y[trainsize:]
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], df2.shape[1]))   # reshape for LSTM input
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], df2.shape[1]))
        self.inputshape = train_X.shape
        return train_X, train_Y, test_X, test_Y



    def compileModel(self, L2, batchSize, dropout, learningRate, optimizer):
        input_shape = self.inputshape
        model = Sequential()                           # LSTM input layer MUST be 3D - (samples, timesteps, features)
        model.add(LSTM(20,
                       return_sequences=True,          # necessary for stacked LSTM layers
                       input_shape=(input_shape[1], input_shape[2])))
        model.add(LSTM(10))
        model.add(Dropout(dropout))
        model.add(Dense(20, kernel_regularizer=l2(L2)))
        model.add(Dropout(dropout))
        model.add(Dense(self.stepsOut))

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[mse])
        return model


    def trainOptimal(self, model, bp, train_X, train_Y):
        print("\tTraining model with optimal parameters...")
        print(bp)
        batch_size = bp['batchSize']
        epochs = bp['epochs']
        history = model.fit(train_X, train_Y,
                                epochs=epochs, batch_size=batch_size,
                                verbose=self.verbose)
        # enforce model to disk (.h5)
        model.save_weights(os.path.join('..', 'models', 'LSTM_best_params.h5'))
        self.history = history
        #self.plotTrainHistory()
        return model

    def gridSearchCV(self, train_X, train_Y):
        print("\tBeggining GridSearchCV to discover optimal hyperparameters")
        model = KerasRegressor(build_fn=self.compileModel, verbose=self.verbose)
        grid = GridSearchCV(estimator=model, param_grid=self.exploreParams, n_jobs=-1, cv=2) #return_train_score=True,
        with joblib.parallel_backend('threading'):
            grid_result = grid.fit(train_X, train_Y)

        bestParameters = grid_result.best_params_
        bestModel = grid_result.best_estimator_.model
        bestModel.summary()
        bestModel.save(os.path.join('..', 'models', 'LSTM_best_params.h5'), overwrite=True)
        print("\tGridSearchCV finished, flashing model to disk and saving best parameters to config...")
        print("\tBest parameters:"+str(bestParameters))

        # Save best found parameters to config file
        BestParamsDict= {}
        BestParamsDict['Optimal LSTM Parameters'] = bestParameters
        with open('config.yaml', 'a') as outfile:
            yaml.dump(BestParamsDict, outfile, default_flow_style=False)
        # history = bestModel.history.history
        return bestModel

    def buildModel(self, train_X, train_Y):
        print("\tBuilding model...")
        if self.train:
            if self.optimExists:                                 # Train using optimal parameters
                bp = self.bestParams
                model = self.compileModel(bp['L2'], bp['batchSize'], bp['dropout'], bp['learningRate'], bp['optimizer'])
                model = self.trainOptimal(model, bp, train_X, train_Y)
            else:                                                # GridSearchCV for hyperparameter tuning
                model = self.gridSearchCV(train_X, train_Y)
        else:
            if os.path.exists(os.path.join('..' ,'models', 'LSTM_best_params.h5')):       # If a saved model exists, load it
                print("\tLoading best model weights from Disk...")
                model = load_model(os.path.join('..' ,'models', 'LSTM_best_params.h5'))
            else:
                if self.optimExists:                                 # Train using optimal parameters
                    bp = self.bestParams
                    model = self.compileModel(bp['L2'], bp['batchSize'], bp['dropout'], bp['learningRate'], bp['optimizer'])
                    model = self.trainOptimal(model, bp, train_X, train_Y)
                else:
                    model = self.gridSearchCV(train_X, train_Y)
        return model


    def plotTrainHistory(self, history):
        print("\tExporting Train History...")
        plt.figure(figsize=(9, 7), dpi=200)
        plt.plot(history.history["loss"], 'darkred', label="Train")
        if 'val_loss' in history:
            plt.plot(history.history["val_loss"], 'darkblue', label="Validation")
        #plt.plot(history.history["val_loss"], 'darkblue', label="Test")
        plt.title("Loss over epoch")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.exportpath, 'LeaningCurves.png'))
        plt.close()


    def evaluatePlot(self, model, df, train_X, train_Y, test_X, test_Y):
        trainPredict = model.predict(train_X)
        testPredict = model.predict(test_X)

        # invert predictions
        scaler = self.scaler
        trainPredict =scaler.inverse_transform(trainPredict)
        train_Y = scaler.inverse_transform([train_Y])
        testPredict = scaler.inverse_transform(testPredict)
        test_Y = scaler.inverse_transform([test_Y])

        test_Y2 = test_Y.reshape(test_Y.shape[2])
        testPredict2 = testPredict.reshape(testPredict.shape[1])
        train_Y2 = train_Y.reshape(train_Y.shape[1],train_Y.shape[2])

        trainScore = mean_absolute_error(train_Y2, trainPredict)
        print('\tTrain Score: %.2f MAE' % trainScore)
        testScore = mean_absolute_error(test_Y2, testPredict2)
        print('\tTest Score: %.2f MAE' % testScore)

        df2 = df[(-self.stepsOut):]
        df2['predictions'] = testPredict2
        rcParams['figure.figsize'] = 12, 6
        plt.plot(df['requests'])
        plt.plot(df2['predictions'], color='brown')
        plt.savefig(os.path.join(self.exportpath, 'LSTM_Forecast.jpg'))
        plt.close()
