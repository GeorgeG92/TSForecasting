import numpy as np
import pandas as pd
import os
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


from datetime import datetime

class SARIMAXModel():
    def __init__(self, df, train, config, exportpath=os.path.join('..', 'output', 'SARIMAX')):
        self.train = train
        self.stepsOut = config['Forecasting']['stepsOut']
        self.exploreParams = config['Forecasting']['SARIMAX']['GridSearchCV']
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        self.exportpath = exportpath
        if 'Optimal SARIMAX Parameters' in config:
            self.optimExists = True
            self.bestParams = config['Optimal SARIMAX Parameters']
        rcParams['figure.figsize'] = 12, 6
        rcParams.update({'figure.autolayout': True})
        matplotlib.rc('font', size=14)
        matplotlib.rc('axes', titlesize=22)
        plt.gcf().autofmt_xdate()

        df.index = pd.date_range(df.index[0], df.index[-1], freq='H')
        print("SARIMAX Modeling")
        self.generatePlots(df)
        self.ModelSARIMAX(df)


    def generatePlots(self, df):
        print("\tExporting ACF/PACF plots...")
        params = {}
        params['s'] = 24     # seasonality is daily ~ 24h
        data =  df['requests']

        lag_acf = acf(data, nlags=50)
        lag_pacf = pacf(data, nlags=50)

        # Plot ACF:
        plt.figure(figsize=(15,5))
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

    def scoreSARIMAX(train, exog_train, params):
        model = SARIMAX(train, order=params[:3], seasonal_order=params[3:], freq='h', exog=exog_train)
        results = model.fit(disp=False, start_ar_lags=13)
        return (params, results.aic)

    def GridSearchSARIMAX(self, train, exog_train):                          # read hp ranges from config
        params = self.exploreParams
        p = params['p']#range(0, 4)
        d = params['d']#range(0, 2)
        q = params['q']#range(0, 4)
        P = params['P']#range(0, 4)
        D = params['D']#range(0, 2)
        Q = params['Q']#range(0, 4)
        s = params['s']#range(24, 25)                                      #
        pdq = list(itertools.product(p, d, q, P, D, Q, s))
        scores = None
        # Full parallel Hyperparameter GridSearchCV
        print("\tStarting GridSearch for best hyperparameters, evaluating "+str(len(pdq))+" models")
        print("\t\tEstimated Time: "+str(len(pdq)*0.4)+" minutes")
        executor = Parallel(n_jobs=-1, backend='threading')
        tasks = (delayed(scoreSARIMAX)(train, exog_train, cfg) for cfg in pdq)
        scores = executor(tasks)
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        scores.sort(key=lambda tup: tup[1])
        print("\tLowest score is "+str(scores[0][1])+", using parameters "+str(scores[0][0]))
        return scores[0][0]

    def trainTestSplit(self, df, data):
        train = data[:-(self.stepsOut)]
        test = data[-(self.stepsOut):]
        exog = df[['Temperature', 'Precipitation', 'WindSpeed']]
        exog_train = exog[:-(self.stepsOut)]
        exog_test = exog[-(self.stepsOut):]
        return train, test, exog_train, exog_test

    def trainPredictSARIMAX(self, df, params=None):
        # Log Transform to battle Heteroscedasticity
        data = np.log(df['requests'])
        # Train/Test Split
        train, test, exog_train, exog_test = self.trainTestSplit(df, data)

        if os.path.exists(os.path.join('..', 'models', 'SARIMAX_best_params.p')):   # if model exists then load, otherwise train on optimal, otherwise GridSearchCV
            print("\tLoading Model from Disk...")
            result = SARIMAXResults.load(os.path.join('..', 'models', 'SARIMAX_best_params.p'))
        elif params:
            print("\tTraining using optimal parameters")
            p,d,q,P,D,Q,s = params['p'], params['d'], params['q'], params['P'], params['D'], params['Q'], params['s']
            model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=exog_train)
            result = model.fit(disp=False)
            if not os.path.exists(os.path.join('..', 'models', 'SARIMAX_best_params.p')):
                result.save(os.path.join('..', 'models', 'SARIMAX_best_params.p'))
        else:
            bp = self.GridSearchSARIMAX(train, exog_train)
            p,d,q,P,D,Q,s = bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6]
            model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=exog_train)
            result = model.fit(disp=False)
            result.save(os.path.join('..', 'models', 'SARIMAX_best_params.p'))
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
        print("\tTrain MAE is: "+str(score))
        score = mean_absolute_error(testRescaled, testpredsRescaled)
        print("\tTest MAE is: "+str(score))

        # Plot
        print("\tExporting Forecast and Residual plots... ")
        plt.figure(figsize=(15,5))
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

        residualPlot = result.plot_diagnostics(figsize = (12, 7), lags=20)
        residualPlot.savefig(os.path.join(self.exportpath, 'residualPlot.png'))


    def ModelSARIMAX(self, df):
        if self.train:
            if self.optimExists:
                self.trainPredictSARIMAX(df, self.bestParams)
            else:
                self.trainPredictSARIMAX(df)
        else:
            if self.optimExists:
                bp = self.bestParams
                self.trainPredictSARIMAX(df, self.bestParams)
            else:
                self.trainPredictSARIMAX(df)
