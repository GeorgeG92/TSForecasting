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
import logging

logger = logging.getLogger(__file__)

class TS_Analysis():
    def __init__(self, args, df):
        logger.info("Multivariate Time Series Analysis")
        self.exportpath = os.path.join(args.exportpath, 'TS_Decomposition')
        self.plot_ts(df)
        self.ts_decomposition(df)
        self.stationarity_tests(df)

    def plot_ts(self, df):
        """ Plots Time Series data and saves it to selected output directory
        Args:
            df: the dataframe with Temperature, Precipitation, WindSpeed columns
        """
        logger.info("\tPlotting TS data and exporting to {path}".format(path=self.exportpath))
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

    def ts_decomposition(self, df):                                   # examine trend, seasonality
        """ Decomposes a Time Series signal to examine trend and seasonality components
            saves the plot in the output directory
        Args:
            df: the dataframe containing time series information 
        """
        rcParams['figure.figsize'] = 18, 8
        logger.info("\tExporting Time Series Decomposition to {path}".format(path=self.exportpath))
        usefulcols = ['requests', 'Temperature', 'Precipitation', 'WindSpeed']
        for col in usefulcols:                           # do it for all relevant TS columns
            temp = df[[col]]
            decomposition = sm.tsa.seasonal_decompose(temp, model='additive')
            fig = decomposition.plot()
            fig.savefig(os.path.join(self.exportpath, str(col)+'_decomposition.png'))
            plt.close(fig)


    def adf_test(self, timeseries):                  
        """ Perform Dickey-Fuller test: The more negative the Test Statistic is, the harder we reject H0: unit root/stationary
            equally: H0: TS is non-stationary
        Args:
            timseries: the timeseries dataframe
        """
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return dfoutput

    def stationarity_tests(self, df):
        """ Checks for stationarity to determine whether its necessary to transform. In principle we do not need to check for 
            stationarity nor correct for it when we are using an LSTM. However, if the data is stationary, it will help with 
            better performance and make it easier for the neural network to learn. For Autoregression models it is mandatory.
        Args:
            df: input dataframe with the time series data
        """
        logger.info("\tTesting if TS is stationary and exporting results to {path}".format(path=os.path.join(self.exportpath,"/ADFtestResults.csv")))
        dftest = pd.DataFrame()
        for col in df.columns:                # return all results for both tests and all TS in a dataframe
            dfadf = self.adf_test(df[col])
            row = pd.Series({'H0 Rejected':1 if dfadf.loc['p-value'] <= 0.05 else 0})   # result of the test
            dfadf = dfadf.append(row)
            dftest[col,'ADF'] = dfadf
        dftest.to_csv(os.path.join(self.exportpath, 'ADFtestResults.csv'))
