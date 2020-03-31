import pandas as pd
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings



class DataLoader():
    def __init__(self, datapath='../data/routes.csv', missingpath="../plots", weatherdata='../data/lima_2015_weatherdata.csv'):
        warnings.filterwarnings("ignore")
        self.datapath = datapath
        assert os.path.exists(self.datapath)
        self.__setmissingpath(missingpath)
        self.__setweatherpath(weatherdata)                     # additional data to enrich the dataset
        df = pd.read_csv("../data/routes.csv", sep='\t')
        df = self.cleanData(df)
        self.__setData(df)
        self.__setResampledData__(self.mergeWeatherData(df))           # after cleaning & merging

    def getData(self):
        return self.__data

    def getResampledData(self):
        return self.__resampledData

    def getmissingpath(self):
        return self.__missingpath

    def getweatherpath(self):
        return self.__weatherdata

    def __setData(self, df):
        self.__data = df

    def __setResampledData__(self, df):
        self.__resampledData = df

    def __setmissingpath(self, path):
        self.__missingpath = path

    def __setweatherpath(self, path):
        self.__weatherdata = path

    def cleanData(self, df):
        print("Cleaning the Data...")
        sns.set(rc={'figure.figsize':(18,14), 'figure.dpi':100})
        sns_plot = sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        bottom, top = sns_plot.get_ylim()                          # fix cropped axis problem
        sns_plot.set_ylim(bottom + 1.0, top - 1.0)

        # if not os.path.exists(self.getmissingpath()):
        #     os.mkdir(self.getmissingpath())
        fig = sns_plot.figure.savefig(self.getmissingpath()+"/missingdata.png")
        #plt.figure(figsize=(14, 11), dpi=200)
        plt.close(fig)
        print("\tWe have "+str(np.around(len(df[df.isnull().any(axis=1)])/len(df)*100, decimals=3))+"% missing values so we drop them")    #def loadWeatherData(self):
        df = df.dropna()
        return df

    def loadWeatherData(self):
        """
        Hourly sampled dataset with temperature, wind and rain information
        """
        print("Loading additional weather data...")
        assert os.path.exists(self.getweatherpath())
        weatherdf = pd.read_csv(self.getweatherpath())
        return weatherdf

    def mergeWeatherData(self, df):
        dfw = self.loadWeatherData()
        #Resample df to a frequency of an hour
        df["request_date"]= pd.to_datetime(df["request_date"])
        df = df.resample('h', on = 'request_date').count()
        df = df.rename(columns={'passenger_id': 'requests'})
        df = df[['requests']]

        # Reset Indexes to merge
        df = df.reset_index()
        dfw = dfw.reset_index()

        dfw['datetime'] = pd.to_datetime(dfw['datetime'])
        df = df.merge(dfw, left_on='request_date', right_on='datetime', how='left')
        df = df.fillna(df.mean())                                        # impute 6 missing values with mean
        df.drop(columns=['datetime', 'index'], inplace=True)                      # remove 2nd index

        # try to reindex
        df = df.set_index('request_date')
        df = df.asfreq(freq='h')
        return df
