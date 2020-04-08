import numpy as np
import pandas as pd
import os
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error


from datetime import datetime

class SARIMAXModel():
    def __init__(self, df, config, exportpath='../output/SARIMAX'):
        self.stepsIn = config['Forecasting']['stepsIn']
        self.stepsOut = config['Forecasting']['stepsOut']
        self.testsize = (self.stepsIn+self.stepsOut)
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        self.exportpath = exportpath
        rcParams['figure.figsize'] = 12, 6
        rcParams.update({'figure.autolayout': True})
        matplotlib.rc('font', size=14)
        matplotlib.rc('axes', titlesize=22)

        df.set_index('request_date', inplace=True)
        print("SARIMAX Modeling")
        data, params = self.estimateParameters(df)
        model = self.trainPredictSARIMAX(data, df, params)

    def difference(self, dataset, interval=1):
        index = list(dataset.index)
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return (diff)

    def adf_test(self, timeseries):
        dftest = adfuller(timeseries, autolag='AIC')   # fail to reject H0: TS is non-stationary - time structure exists
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        #print (dfoutput)
        return dfoutput

    def makeStationary(self, df):
        d=-1
        temp = df
        for i in range(10):
            pvalue = self.adf_test(temp)[1]           # stationary
            if pvalue<=0.05:
                d=i
                return temp, d
            else:
                temp = difference(temp,i+1)
        if d==-1:
            print("Error: Time Series differenced 10 times but is still not stationary")
            return -1, -1


    def estimateParameters(self, df):
        print("\tEstimating model parameters...")
        params = {}
        params['s'] = 24     # seasonality is daily ~ 24h
        data, params['d'] = self.makeStationary(df['requests'])    # difference

        # TS decomposition - Plot

        additive = seasonal_decompose(data, model='additive', freq=params['s'])    # freq is every how many dps we observe the same shit
        trend, seasonal, residual = additive.trend, additive.seasonal, additive.resid
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(data, '-')
        axs[0].set_title('Time Series')
        axs[0].set_ylabel('Requests')

        axs[1].plot(additive.trend, '-')
        axs[1].set_title('Trend')
        axs[1].set_ylabel('Requests')

        axs[2].plot(additive.seasonal, '--')
        axs[2].set_title('Seasonality')
        axs[2].set_ylabel('Requests')

        axs[3].plot(additive.resid, '--')
        axs[3].set_title('Residuals')
        axs[3].set_ylabel('Requests')

        #fig.savefig(self.exportpath+'/TS_Decomposition.jpeg')

        # PACF plot

        #pacf_plot = plot_pacf(data, lags=30)
        #pacf_plot.savefig(self.exportpath+'/PACF_Plot.jpeg')
        # choose p: params['p'] =

        # ACF plots

        #acf_plot = plot_acf(data, lags=30)
        #acf_plot.savefig(self.exportpath+'/ACF_Plot.jpeg')
        # choose q: params['q'] =

        useless, params['D'] = self.makeStationary(seasonal)

        # PACF plot for Seasonality

        #pacf_plot = plot_pacf(seasonal, lags=30)
        #pacf_plot.savefig(self.exportpath+'/Seasonality_PACF_Plot.jpeg')
        # choose P: params['P'] =

        # PACF plot for Seasonality

        #pacf_plot = plot_acf(seasonal, lags=30)
        #pacf_plot.savefig(self.exportpath+'/Seasonality_ACF_Plot.jpeg')
        # choose Q: params['Q'] =

        params['p'] = 2
        params['q'] = 5
        useless, params['D'] = self.makeStationary(seasonal)
        params['P'] = 2
        params['Q'] = 2
        params['s'] = 24

        return data, params

    def trainPredictSARIMAX(self, data, df, params):
        print("\tTraining the model...")
        rcParams['figure.figsize'] = 14, 8

        train = data[:-(self.stepsOut)]
        test = data[-(self.stepsOut):]

        myCols = [col for col in df.columns if col!='requests']
        #print(myCols)
        exog = df[myCols]
        exog_train = df[:-(self.stepsOut)]
        exog_test = df[-(self.stepsOut):]

        # Fit
        p,d,q,P,D,Q,s = params['p'], params['d'], params['q'], params['P'], \
            params['D'], params['Q'], params['s']
        #model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=exog_train)
        model = ARIMA(train, order=(p,d,q))
        resultfit = model.fit(disp=False)
        #print(result.summary())

        # Evaluate
        trainpreds = resultfit.fittedvalues
        testpreds = resultfit.predict(start=test.index[0], end=test.index[0-1])#, exog=exog_test)
        score = mean_absolute_error(train, trainpreds)
        print("\tTrain MAE is: "+str(score))
        score = mean_absolute_error(test, testpreds)
        print("\tTest MAE is: "+str(score))

        print(test.index[0])
        print(type(test.index[0]))
        plt.clf()
        # Plot Forecast
        plt.plot(data, color='deepskyblue')
        plt.plot(trainpreds, color='navy')
        plt.plot(testpreds, color='lightcoral')
        #plt.axvline(test.index[0], label='train/test split', c='mediumvioletred')
        plt.savefig(self.exportpath+"/SARIMAX_fit.png")
        plt.close()
