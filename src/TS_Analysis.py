import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import os
from pylab import rcParams
import seaborn as sns
import warnings


class TS_Analysis():
    def __init__(self, df, exportpath=os.path.join('..', 'output', 'TS_Decomposition')):
        print("Multivariate Time Series Analysis")
        warnings.filterwarnings("ignore")
        self.exportpath = exportpath
        self.plotTS(df)
        self.TSDecomposition(df)
        self.stationarityTests(df)

    def plotTS(self, df):
        print("\tPlotting TS data and exporting to "+str(self.exportpath))
        if not os.path.exists(self.exportpath):
            os.mkdir(self.exportpath)
        #df.reset_index(inplace=True)
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15,15))
        figure = sns.lineplot(df.index, y='requests', color="indianred", data=df, ax=axs[0])
        for item in figure.get_xticklabels():
            item.set_rotation(45)
        figure = sns.lineplot(df.index, y='Temperature', color="blue", data=df, ax=axs[1])
        for item in figure.get_xticklabels():
            item.set_rotation(45)
        figure = sns.lineplot(df.index, y='Precipitation', color="green", data=df, ax=axs[2])
        for item in figure.get_xticklabels():
            item.set_rotation(45)
        figure = sns.lineplot(df.index, y='WindSpeed', color="black", data=df, ax=axs[3])
        for item in figure.get_xticklabels():
            item.set_rotation(45)
        plt.savefig(os.path.join(self.exportpath, 'plots.png'))
        plt.close(fig)

    def TSDecomposition(self, df):                                   # examine trend, seasonality
        rcParams['figure.figsize'] = 18, 8
        print("\tExporting Time Series Decomposition to "+str(self.exportpath))
        usefulcols = ['requests', 'Temperature', 'Precipitation', 'WindSpeed']
        for col in usefulcols:                           # do it for all relevant TS columns
            temp = df[[col]]
            #temp = temp.set_index('request_date')
            #temp = temp.asfreq(freq='h')

            decomposition = sm.tsa.seasonal_decompose(temp, model='additive')
            fig = decomposition.plot()
            fig.savefig(os.path.join(self.exportpath, str(col)+'_decomposition.png'))
            plt.close(fig)


    def adf_test(self, timeseries):                  # The more negative the Test Statistic is, the harder we reject H0: unit root/stationary
        #Perform Dickey-Fuller test:                 # equally: H0: TS is non-stationary
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return dfoutput

    def stationarityTests(self, df):        # check for stationarity to determine whether its necessary to transform
        # In principle we do not need to check for stationarity nor correct for it when we are using an LSTM. However,
        # if the data is stationary, it will help with better performance and make it easier for the neural network
        # to learn.
        print("\tTesting if TS is stationary and exporting results to "+str(self.exportpath)+"/ADFtestResults.csv")
        dftest = pd.DataFrame()
        for col in df.columns:                # return all results for both tests and all TS in a dataframe
            dfadf = self.adf_test(df[col])
            row = pd.Series({'H0 Rejected':1 if dfadf.loc['p-value'] <= 0.05 else 0})   # result of the test
            dfadf = dfadf.append(row)
            dftest[col,'ADF'] = dfadf
        dftest.to_csv(os.path.join(self.exportpath, 'ADFtestResults.csv'))
