import numpy as np
import pandas as pd
import os
import statsmodels as sm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SARIMAXModel():
    def __init__(self, df, config, exportpath='../output/SARIMAX'):
        self.stepsIn = config['Forecasting']['stepsIn']
        self.stepsOut = config['Forecasting']['stepsOut']
        self.testsize = (self.stepsIn+self.stepsOut)
        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        self.exportpath = exportpath
        rcParams['figure.figsize'] = 12, 6
        df.set_index('request_date', inplace=True)

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
        params = {}
        data, params['d'] = self.makeStationary(df['requests'])    # difference

        # TS decomposition - Plot

        additive = seasonal_decompose(data, model='additive', freq=24)    # freq is every how many dps we observe the same shit
        trend, seasonal, residual = additive.trend, additive.seasonal, additive.resid

        # Extract Parameters from plots

        params['p'] = 2
        params['q'] = 5
        useless, params['D'] = self.makeStationary(seasonal)
        params['P'] = 2
        params['Q'] = 2
        params['s'] = 24

        return data, params

    def trainPredictSARIMAX(self, data, df, params):
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
        model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=exog_train)
        resultfit = model.fit(disp=False)
        #print(result.summary())

        # Evaluate
        trainpreds = resultfit.fittedvalues
        testpreds = resultfit.predict(start=test.index[0], end=test.index[0-1], exog=exog_test)
        score = mean_absolute_error(test, testpreds)
        print("MAE is: "+str(score))

        # Plot Forecast
        rcParams['figure.figsize'] = 14, 8
        plt.plot(data, color='deepskyblue')
        plt.plot(trainpreds, color='navy')
        plt.plot(testpreds, color='lightcoral')
        plt.axvline(x=test.index[0], label='train/test split', c='mediumvioletred')
        plt.savefig(self.exportpath+"/SARIMAX_fit.png")
        plt.close()
