import pandas as pd
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings

class DataLoader():
    def __init__(self, datapath, imputemode, exportpath="../output/Cleaning", weatherpath='../data/lima_2015_weatherdata.csv'):
        warnings.filterwarnings("ignore")
        df = pd.read_csv(datapath, sep='\t')
        df = self.cleanData(df, imputemode, exportpath)
        df = self.mergeWeatherData(df, weatherpath)
        self.data = df
        #save df


    def getData(self):
        return self.data

    def impute(self, df, imputemode):           # 2 be implemented
        return df
        # if imputemode=='fast':
        #     # KNN
        #     sa
        # else:
        #     # MICE

    def cleanData(self, df, imputemode, exportpath):
        print("Cleaning the Data...")
        sns.set(rc={'figure.figsize':(18,14), 'figure.dpi':50})
        sns.set(font_scale=2.2)
        matplotlib.rcParams.update({'figure.autolayout': True})

        sns_plot = sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        bottom, top = sns_plot.get_ylim()                          # fix cropped axis problem
        sns_plot.set_ylim(bottom + 1.0, top - 1.0)

        if not os.path.exists(exportpath):
            os.mkdir(exportpath)
        fig = sns_plot.figure.savefig(exportpath+"/missingdata.png")
        #plt.figure(figsize=(14, 11), dpi=200)
        plt.close(fig)

        # Drop columns with > threshold missing values
        colNanThreshold = 20   # %
        rowNanThreshold = 0.05 #
        missing = pd.DataFrame((df.isna().mean().round(4) * 100), columns=['percentage']).reset_index()
        dropcols = list(missing[missing['percentage']>=colNanThreshold]['index'])

        if len(dropcols):
            print('\tColumns '+str(dropcols)+" have more than "+str(colNanThreshold)+"% nan values, thus get dropped entirely")
            df = df.drop(columns=dropcols)

        # Drop NaN rows based on columns of interest
        uselessCols = ['ID', 'Name', 'Photo', 'Nationality', 'Flag', 'Club', 'Club Logo', 'Wage', 'Work Rate', 'Body Type', 'Real Face']
        importantCols = [col for col in df.columns if col not in uselessCols]
        missingPerc = np.around(len(df[df[importantCols].isnull().any(axis=1)])/len(df)*100, decimals=3)
        if missingPerc<rowNanThreshold:
            print("\tWe have "+str(missingPerc)+"% missing values so we drop them")    #def loadWeatherData(self):
            df = df.dropna()
        else:
            print("\tWe have more than "+str(missingPerc)+"% missing values, attempting imputation")
            df = self.impute(df, imputemode)
        return df

    def loadWeatherData(self, weatherpath):
        """
        Hourly sampled dataset with temperature, wind and rain information
        """
        print("Loading additional weather data...")
        assert os.path.exists(weatherpath)
        weatherdf = pd.read_csv(weatherpath)
        return weatherdf

    def mergeWeatherData(self, df, weatherpath):
        dfw = self.loadWeatherData(weatherpath)
        #Resample df to a frequency of an hour
        df["request_date"]= pd.to_datetime(df["request_date"])
        df = df.resample('h', on = 'request_date').count()              # read frequency from config
        df = df.rename(columns={'passenger_id': 'requests'})
        df = df[['requests']]

        # Reset Indexes to merge
        df = df.reset_index()
        dfw = dfw.reset_index()

        dfw['datetime'] = pd.to_datetime(dfw['datetime'])
        df = df.merge(dfw, left_on='request_date', right_on='datetime', how='left')
        df = df.fillna(df.mean())                                                 # impute 6 missing values with mean
        df.drop(columns=['datetime', 'index'], inplace=True)                      # remove 2nd index

        # try to reindex
        df = df.set_index('request_date')
        df = df.asfreq(freq='h')
        return df
