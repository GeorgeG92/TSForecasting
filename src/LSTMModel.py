from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
from keras.regularizers import l2
from keras.metrics import mse
import pandas as pd
import numpy as np
import seaborn as sns
import math
import os
import yaml
import warnings


class LSTMModel():
    def __init__(self, df, train, config, exportpath='../output/LSTM'):
        self.train = train
        self.verbose = config['Forecasting']['verbose']
        if 'Optimal Parameters' in config:
            self.optimExists = True
            self.bestParams = config['Optimal Parameters']
        else:
            self.optimExists = False
            self.exploreParams = config['Forecasting']['LSTM']['GridSearchCV']
        self.testsize = config['Forecasting']['testSize']
        self.stepsIn = config['Forecasting']['stepsIn']
        self.stepsOut = config['Forecasting']['stepsOut']
        self.gscvDict = config['Forecasting']['LSTM']
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        self.exportpath = exportpath
        #self.plotTimeSeries(df)
        train_X, train_Y, test_X, test_Y = self.preProcess(df)
        model = self.buildModel(train_X, train_Y)
        #self.evaluatePlot(model, df, train_X, train_Y, test_X, test_Y)


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
        scaler = StandardScaler()
        df2 = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)
        self.scaler = scaler
        data = np.array(df2)
        X, Y = split_sequences(data, stepsIn, stepsOut)

        # Train/set split
        trainsize = int((1-testSize)*X.shape[0])             # 85/15 split
        train_X = X[:trainsize, :]
        train_Y = Y[:trainsize]
        test_X = X[trainsize:, :]
        test_Y = Y[trainsize:]
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], df2.shape[1]))   # reshape for LSTM input
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], df2.shape[1]))
        self.inputshape = train_X.shape
        return train_X, train_Y, test_X, test_Y



    def compileModel(self, L2, batchSize, dropout, learningRate, optimizer):
        # LSTM model 1
        input_shape = self.inputshape
        model = Sequential()                           # LSTM input layer MUST be 3D - (samples, timesteps, features)
        model.add(LSTM(50, activation='relu',
                       return_sequences=True,          # necessary for stacked LSTM layers
                       kernel_regularizer=l2(L2),
                       input_shape=(input_shape[1], input_shape[2])))
        model.add(LSTM(50, kernel_regularizer=l2(L2), activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(20, kernel_regularizer=l2(L2)))
        model.add(Dropout(dropout))
        model.add(Dense(self.stepsOut))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[mse])
        return model


    def trainOptimal(self, model, params, train_X, train_Y):
        print("Training model with optimal parameters...")
        batch_size = bp['batchSize']
        epochs = bp['epochs']
        model = self.model
        history = model.fit(train_X, train_Y,
                                epochs=epochs, batch_size=batch_size,
                                verbose=1)
        # enforce model to disk (.h5)
        model.save_weights('../model/LSTM_best_params.h5')
        self.history = history
        self.plotTrainHistory()
        return model

    def gridSearchCV(self, train_X, train_Y):
        print(train_X.shape, train_Y.shape)
        print(self.exploreParams)
        model = KerasRegressor(build_fn=self.compileModel, verbose=self.verbose)
        grid = GridSearchCV(estimator=model, param_grid=self.exploreParams, n_jobs=-1, return_train_score=True, cv=2)
        grid_result = grid.fit(train_X, train_Y)

        bestParameters = grid_result.best_params_
        bestModel = grid_result.best_estimator_.model
        bestModel.summary()
        bestModel.save('../model/LSTM_best_params.h5', overwrite=True)
        print("GridSearchCV finished, flashing model to disk and saving best parameters to config...")
        print("Best parameters:"+str(bestParameters))

        # Save best found parameters to config file
        BestParamsDict= {}
        BestParamsDict['Optimal Parameters'] = bestParameters
        with open('config.yaml', 'a') as outfile:
            yaml.dump(BestParamsDict, outfile, default_flow_style=False)
        # plot history
        # history = bestModel.history.history
        return bestModel

    def buildModel(self, train_X, train_Y):
        print("Building model...")
        if self.train:
            if self.optimExists:                                 # Train using optimal parameters
                bp = self.bestParams
                model = self.compileModel(bp['L2'], bp['batchSize'], bp['dropout'], bp['learningRate'], bp['optimizer'])
                model = self.trainOptimal(model, bp, train_X, train_Y)
            else:                                                # GridSearchCV for hyperparameter tuning
                print("Beggining GridSearchCV to discover optimal hyperparameters")
                model = self.gridSearchCV(train_X, train_Y)
        else:
            if os.path.exists('../model/LSTM_best_params.h5'):       # If a saved model exists, load it
                print("Loading model weights from Disk...")
                model = load_model('../model/LSTM_best_params.h5')
            else:
                print("No saved model detected, forcing Training")
                model = self.gridSearchCV(train_X, train_Y)
        return model


    def plotTrainHistory(self, history):
        print("Exporting Train History...")
        plt.figure(figsize=(9, 7), dpi=200)
        plt.plot(history.history["loss"], 'darkred', label="Train")
        if 'val_loss' in history:
            plt.plot(history.history["val_loss"], 'darkblue', label="Validation")
        #plt.plot(history.history["val_loss"], 'darkblue', label="Test")
        plt.title("Loss over epoch")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.exportpath+'./LeaningCurves.png')
        plt.close()


    def evaluatePlot(self, model, df, train_X, train_Y, test_X, test_Y):
        train_X_array = train_X
        test_X_array = test_X
        scaler = self.scaler

        trainPredict = np.array(model.predict(train_X_array))         # forecast based on training data
        testPredict = np.array(model.predict(test_X_array))           # forecast of test (unseen data)

        print(test_X_array.shape)
        test_X_array = test_X_array.reshape((test_X_array.shape[0], test_X_array.shape[2]))

        # invert scale for train forecast (train)
        train_X_array = train_X_array.reshape((train_X_array.shape[0], train_X_array.shape[2]))
        trainPredict = trainPredict.reshape((len(trainPredict), 1))
        inv_x = np.concatenate((trainPredict, train_X_array[:, 1:]), axis=1)
        inv_x = scaler.inverse_transform(inv_x)
        inv_x = inv_x[:,0]

        # invert scaling for forecast (test)
        inv_yhat = np.concatenate((testPredict, test_X_array[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        # invert scaling for actual values
        test_Y_array = test_Y
        test_Y_array = test_Y_array.reshape((len(test_Y_array), 1))
        inv_y = np.concatenate((test_Y_array, test_X_array[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]


        # calculate RMSE based on original-scaled data
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('\tPerformance (RMSE) on Test Data is : %.3f' % rmse)


        ### Merge inv_y with train_Y_pred!
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
