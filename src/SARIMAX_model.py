import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import logging

logger = logging.getLogger(__file__)


class SARIMAXModel():
    """ Implements the SARIMAX model class 
    """
    def __init__(self, args, config, df):
        self.train = args.train
        self.stepsOut = config['Forecasting']['stepsOut']
        self.exploreParams = config['Forecasting']['SARIMAX']['GridSearchCV']
        self.exportpath = os.path.join(args.exportpath, 'SARIMAX')
        self.modelpath = os.path.join(args.modelpath, 'SARIMAX_best_params.p')
        if 'Optimal SARIMAX Parameters' in config:
            self.optimExists = True
            self.bestParams = config['Optimal SARIMAX Parameters']
        rcParams['figure.figsize'] = 12, 6
        rcParams.update({'figure.autolayout': True})
        matplotlib.rc('font', size=14)
        matplotlib.rc('axes', titlesize=22)
        plt.gcf().autofmt_xdate()

        self.df = df
        self.df.index = pd.date_range(df.index[0], df.index[-1], freq='H')


    def generate_plots(self, df):
        """ Generates the ACF/PACF plots for the timeseries
        Args:
            df: the dataframe containing the time series data
        """
        logger.info("\tExporting ACF/PACF plots...")
        params = {}
        params['s'] = 24     # seasonality is daily ~ 24h
        data =  df['requests']

        lag_acf = acf(data, nlags=50)
        lag_pacf = pacf(data, nlags=50)

        # Plot ACF:
        plt.figure(figsize=(15,7))
        plt.subplot(121)
        plt.stem(lag_acf)
        plt.axhline(y=0, linestyle='-', color='black')
        plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        plt.xlabel('Lag')
        plt.ylabel('ACF')

        # Plot PACF:
        plt.subplot(122)
        plt.stem(lag_pacf)
        plt.axhline(y=0, linestyle='-', color='black')
        plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
        plt.xlabel('Lag')
        plt.ylabel('PACF')
        plt.savefig(os.path.join(self.exportpath, 'ACF_PACF_Plot.jpeg'))

    def score_sarimax(train, exog_train, params):
        """ Trains a SARIMAX model using given parameters
        Args:
            train: the data to train to
            exog_train: exogenous features for the time series
            params: a dictionary of parameters to train the model with
        """
        model = SARIMAX(train, order=params[:3], seasonal_order=params[3:], freq='h', exog=exog_train)
        results = model.fit(disp=False, start_ar_lags=13)
        return (params, results.aic)

    def grid_search_sarimax(self, train, exog_train):
        """ Performs GridSearch on SARIMAX training to determine the best model hyperaparameters
        Args:
            train: the data to train on
            exog_train: exogenous features besides the main time series feature
        """
        params = self.exploreParams
        p = params['p']#range(0, 4)
        d = params['d']#range(0, 2)
        q = params['q']#range(0, 4)
        P = params['P']#range(0, 4)
        D = params['D']#range(0, 2)
        Q = params['Q']#range(0, 4)
        s = params['s']#range(24, 25)                                      
        pdq = list(itertools.product(p, d, q, P, D, Q, s))
        scores = None
        # Full parallel Hyperparameter GridSearchCV
        logger.info("\tStarting GridSearch for best hyperparameters, evaluating {num} models".format(num=len(pdq)))
        logger.info("\t\tEstimated Time: {num} mminutes".format(num=len(pdq)*0.4))
        executor = Parallel(n_jobs=-1, backend='threading')
        tasks = (delayed(score_sarimax)(train, exog_train, cfg) for cfg in pdq)
        scores = executor(tasks)
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        scores.sort(key=lambda tup: tup[1])
        logger.info("\tLowest score is {score}, using parameters {params}".format(score=scores[0][1], params=scores[0][0]))
        return scores[0][0]

    def train_test_split(self, df, data):
        """ Splits data/exogenous data into train/test slits
        Args:
            df: the dataframe with all the features (incl. exogenous)
            data: the main timeseries data
        """
        train = data[:-(self.stepsOut)]
        test = data[-(self.stepsOut):]
        exog = df[['Temperature', 'Precipitation', 'WindSpeed']]
        exog_train = exog[:-(self.stepsOut)]
        exog_test = exog[-(self.stepsOut):]
        return train, test, exog_train, exog_test

    def train_predict_sarimax(self, df, params=None):
        """ Trains a SARIMAX model on the train data and runs inference on the test split
        Args:
            df: the dataframe containing the data
            params: optional argument containing the best discovered parameters for the model
        """

        # Log Transform to battle Heteroscedasticity
        data = np.log(df['requests'])
        # Train/Test Split
        train, test, exog_train, exog_test = self.train_test_split(df, data)

        if os.path.exists(self.modelpath):      # if model exists then load, otherwise train on optimal, otherwise GridSearchCV
            logger.info("\tLoading Model from Disk...")
            result = SARIMAXResults.load(self.modelpath)
        elif params:
            logger.info("\tTraining using optimal parameters")
            p,d,q,P,D,Q,s = params['p'], params['d'], params['q'], params['P'], params['D'], params['Q'], params['s']
            model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=exog_train)
            result = model.fit(disp=False)
            if not os.path.exists(self.modelpath):
                result.save(self.modelpath)
        else:
            bp = self.grid_search_sarimax(train, exog_train)
            p,d,q,P,D,Q,s = bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6]
            model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=exog_train)
            result = model.fit(disp=False)
            result.save(self.modelpath)
            # Save optimal parameters to config
            # BestParamsDict = {}
            # BestParamsDict['Optimal Parameters'] = bestParameters
            # with open('config.yaml', 'a') as outfile:
            #     yaml.dump(BestParamsDict, outfile, default_flow_style=False)

        #print(result.summary())
        trainpreds = result.fittedvalues
        testpreds = result.predict(start=test.index[0], end=test.index[0-1], exog=exog_test)
        ci = result.get_prediction(start=data.index[len(train)], end=data.index[len(data)-1] , exog=exog_test).conf_int()
        low = ci['lower requests']
        up = ci['upper requests']

        # Inverse scale
        dataRescaled = np.exp(data)
        trainRescaled = np.exp(train)
        testRescaled = np.exp(test)
        trainpredsRescaled = np.exp(trainpreds)
        testpredsRescaled = np.exp(testpreds)
        lowRestored = np.exp(low)
        upRestored = np.exp(up)

        # Evaluate
        score = mean_absolute_error(trainRescaled, trainpredsRescaled)
        logger.info("\tTrain MAE is {mae}".format(mae=score))
        score = mean_absolute_error(testRescaled, testpredsRescaled)
        logger.info("\tTest MAE is {mae}".format(mae=score))

        # Plot
        logger.info("\tExporting Forecast and Residual plots... ")
        plt.figure(figsize=(15,7))
        plt.subplot(211)
        plt.plot(dataRescaled, color='deepskyblue')
        plt.plot(trainpredsRescaled, color='navy')
        plt.plot(testpredsRescaled, color='lightcoral')
        plt.title('TS')
        plt.axvline(x=test.index[0], label='train/test split', c='mediumvioletred')
        plt.subplot(212)
        plt.plot(testpredsRescaled, color='lightcoral')
        plt.plot(testRescaled, color='deepskyblue')
        plt.fill_between(test.index, lowRestored, upRestored, color='pink')
        plt.title('Forecast')
        plt.savefig(os.path.join(self.exportpath, 'SARIMAX_Forecast.png'))
        plt.close()

        residualPlot = result.plot_diagnostics(figsize = (15, 8), lags=20)
        residualPlot.savefig(os.path.join(self.exportpath, 'residualPlot.png'))


    def model_sarimax(self):
        """ Responsible for the train/inference pipeline of the model
        Args:
            df: the dataframe containing the time series data
        """
        logger.info("SARIMAX Modeling")
        self.generate_plots(self.df)
        if self.train:
            if self.optimExists:
                self.train_predict_sarimax(self.df, self.bestParams)
            else:
                self.train_predict_sarimax(self.df)
        else:
            if self.optimExists:
                self.train_predict_sarimax(self.df, self.bestParams)
            else:
                self.train_predict_sarimax(self.df)
