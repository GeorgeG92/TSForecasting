from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR           # suppress tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)     # run before importing tensorflow

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import mse
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

logger = logging.getLogger(__file__)

class LSTMModel():
    """  This class implements the code for the LSTM Neural Network
    """
    def __init__(self, args, config, df):
        self.train = args.train
        self.verbose = config['Forecasting']['verbose']
        if 'Optimal LSTM Parameters' in config:
            self.optimExists = True
            self.bestParams = config['Optimal LSTM Parameters']
        else:
            self.optimExists = False
        self.exploreParams = config['Forecasting']['LSTM']['grid_search_cv']
        self.stepsIn = config['Forecasting']['stepsIn']
        self.stepsOut = config['Forecasting']['stepsOut']
        self.gscvDict = config['Forecasting']['LSTM']
        self.testsize = (self.stepsIn+self.stepsOut)
        self.exportpath = os.path.join(args.exportpath, 'LSTM')
        if not os.path.exists(self.exportpath):
            os.mkdir(self.exportpath)
        self.modelpath = os.path.join(args.modelpath, 'LSTM_best_params.h5')
        logger.info("LSTM Modeling")
        #self.plot_time_series(df)
        train_X, train_Y, test_X, test_Y = self.pre_process(df)
        model = self.build_model(train_X, train_Y)
        self.evaluate_plot(model, df, train_X, train_Y, test_X, test_Y)


    def plot_time_series(self, df):
        """ Plots time series data
        Args:
            df: the dataframe containing the time series data
        """
        train_size = int(len(df)-self.testsize)
        train, test = df[0:train_size], df[train_size:len(df)]
        plt.figure(figsize=(14, 14), dpi=200)
        plt.plot(train['requests'])
        plt.plot(test['requests'])
        plt.xticks(rotation=45)
        plt.savefig(os.path.exists(self.exportpath, "initialplots.png"))
        plt.close()

    def pre_process(self, df):
        """ Performs all the required preprocessing (splitting, casting, scaling)
            for the model run
        Args:
            df: the dataframe containing the time series data
        """
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

        computeCols = [col for col in df.columns if col!='request_date']
        df[computeCols] = df[computeCols].astype('float32')
        #df2 = df[['requests', 'Temperature', 'Precipitation', 'WindSpeed']]

        # Scale & Formulate as a Supervised  Learning method
        scalers = []
        df2 = pd.DataFrame(columns = df.columns, index=df.index)
        for col in df.columns:
            scaler = StandardScaler()
            df2[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1))
            scalers.append(scaler)

        self.scaler = scalers[0]
        data = np.array(df2)
        X, Y = split_sequences(data, self.stepsIn, self.stepsOut)

        # Train/set split
        trainsize = len(df2)-self.testsize
        train_X = X[:trainsize, :]
        train_Y = Y[:trainsize]
        test_X = X[trainsize:, :]
        test_Y = Y[trainsize:]
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], df2.shape[1]))   # reshape for LSTM input
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], df2.shape[1]))
        self.inputshape = train_X.shape
        return train_X, train_Y, test_X, test_Y



    def compile_model(self, L2, batchSize, dropout, learningRate, optimizer):
        """ Compiles the model given the input hyperparameters
        Args:
            L2: l2 regularization parameter
            batchSize: the batch size of the model
            dropout: the dropout rate 
            learningRate: the learning rate parameter
            optimizer: which optimizer to use for model training
        """
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


    def train_optimal(self, model, bp, train_X, train_Y):
        """ Trains the model given a set of optimal hyperparameters
        Args:
            model: a keras model instance
            bp: a dictionary containing the best hyperaparameters
            train_X: the train data features
            train_Y: the train data labels
        """
        logger.info("\tTraining model with optimal parameters...")
        batch_size = bp['batchSize']
        epochs = bp['epochs']
        history = model.fit(train_X, train_Y,
                                epochs=epochs, batch_size=batch_size,
                                verbose=self.verbose)
        # enforce model to disk (.h5)
        model.save(self.modelpath)
        self.history = history
        #self.plot_train_history()
        return model

    def grid_search_cv(self, train_X, train_Y):
        """ Performs grid search cross validation on the train data
        Args:
            train_X: the train data features
            train_Y: the train data labels
        """
        logger.info("\tBeggining grid_search_cv to discover optimal hyperparameters")
        model = KerasRegressor(build_fn=self.compile_model, verbose=self.verbose)
        grid = grid_search_cv(estimator=model, param_grid=self.exploreParams, n_jobs=-1, cv=2) #return_train_score=True,
        grid_result = grid.fit(train_X, train_Y)

        bestParameters = grid_result.best_params_
        bestModel = grid_result.best_estimator_.model
        bestModel.summary()
        bestModel.save(os.path.join(self.modelpath), overwrite=True)
        logger.info("\tgrid_search_cv finished, flashing model to disk and saving best parameters to config...")
        logger.info("\tBest parameters: {params}".format(bestParameters))

        # Save best found parameters to config file
        BestParamsDict= {}
        BestParamsDict['Optimal LSTM Parameters'] = bestParameters
        with open('config.yaml', 'a') as outfile:
            yaml.dump(BestParamsDict, outfile, default_flow_style=False)
        # history = bestModel.history.history
        return bestModel

    def build_model(self, train_X, train_Y):
        """ Builds the Keras model and trains it or loads it from disk
        Args:
            train_X: the train data features
            train_Y: the train data labels
        """
        logger.info("\tBuilding model...")
        if self.train:
            if self.optimExists:                                 # Train using optimal parameters
                bp = self.bestParams
                model = self.compile_model(bp['L2'], bp['batchSize'], bp['dropout'], bp['learningRate'], bp['optimizer'])
                model = self.train_optimal(model, bp, train_X, train_Y)
            else:                                                # grid_search_cv for hyperparameter tuning
                model = self.grid_search_cv(train_X, train_Y)
        else:
            if os.path.exists(self.modelpath):       # If a saved model exists, load it
                logger.info("\tLoading best model weights from Disk...")
                model = load_model(self.modelpath)
            else:
                if self.optimExists:                                 # Train using optimal parameters
                    bp = self.bestParams
                    model = self.compile_model(bp['L2'], bp['batchSize'], bp['dropout'], bp['learningRate'], bp['optimizer'])
                    model = self.train_optimal(model, bp, train_X, train_Y)
                else:
                    model = self.grid_search_cv(train_X, train_Y)
        return model


    def plot_train_history(self, history):
        """ Exports graphs of train loss
        Args:
            history: a history object returned from the Keras training session
        """
        logger.info("\tExporting Train History...")
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


    def evaluate_plot(self, model, df, train_X, train_Y, test_X, test_Y):
        """ Run Inference and persist plotted results to disk 
        Args:
            model: the model to run inference from
            df: the dataframe containing input data
            train_X: the train data features
            train_Y: the train data labels
            test_X: the test data features
            test_Y: the test data labels
        """
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
        logger.info('\tTrain Score: %.2f MAE' % trainScore)
        testScore = mean_absolute_error(test_Y2, testPredict2)
        logger.info('\tTest Score: %.2f MAE' % testScore)

        df2 = df[(-self.stepsOut):]
        df2['predictions'] = testPredict2
        rcParams['figure.figsize'] = 12, 6
        plt.plot(df['requests'])
        plt.plot(df2['predictions'], color='brown')
        plt.savefig(os.path.join(self.exportpath, 'LSTM_Forecast.png'))
        plt.close()
