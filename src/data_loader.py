import pandas as pd
pd.options.mode.chained_assignment = None
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__file__)


class DataLoader():
    """ This Class is reponsible for loading, enriching and preprocessing the data prior to 
        the modeling part
    """
    def __init__(self, args):
        logger.info("Loading and cleaning the Data")
        self.datapath = args.datapath
        self.filename = args.filename
        self.extfilename = args.weatherfile
        self.filepath = os.path.join(self.datapath, self.filename)
        self.extfilepath = os.path.join(self.datapath, self.extfilename)
        self.inputemethod = args.inputemethod
        self.exportpath = os.path.join(args.exportpath, 'Cleaning')
        self.weathercols = ['Temperature', 'Precipitation', 'WindSpeed']

        self.origData = pd.read_csv(self.filepath, sep='\t')
        self.datasize = len(self.origData)
        if self.datasize<50:
            logger.warning("\tWarning: StepsIn + StepsOut should sum to less than 50% of the dataframe size")


    def process_data(self):
        self.data = self.clean_data(self.origData)
        self.data = self.merge_weather_data(self.data)
        return self.data, self.origData

    def impute(self, df):           
        """ to be implemented"""
        # if self.inputemethod=='fast':
        # KNN
        return df.dropna()

    def clean_data(self, df):
        """
        Checks the data for missing values and imputes/cleans them based on method defined in 
            execution arguments
        Args:
            df: the pandas dataframe containing the data
        Returns:
            a dataframe of the cleaned/imputed dataset
        """
        # Visualize missing values and export plot
        sns.set(rc={'figure.figsize':(18,14), 'figure.dpi':50})
        sns.set(font_scale=2.2)
        matplotlib.rcParams.update({'figure.autolayout': True})

        sns_plot = sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        bottom, top = sns_plot.get_ylim()                          # fix cropped axis problem
        sns_plot.set_ylim(bottom + 1.0, top - 1.0)

        fig = sns_plot.figure.savefig(os.path.join(self.exportpath+"missingdata.png"))
        plt.close(fig)

        # Drop columns with > threshold missing values
        colNanThreshold = 20   # %
        rowNanThreshold = 0.05
        missing = pd.DataFrame((df.isna().mean().round(4) * 100), columns=['percentage']).reset_index()
        dropcols = list(missing[missing['percentage']>=colNanThreshold]['index'])
        if len(dropcols):
            logger.info("\tColumns {d} have more than {th}% nan values, thus get dropped entirely".format(d=dropcols, th=colNanThreshold))
            df = df.drop(columns=dropcols)

        # Drop NaN rows based on columns useful for forecasting - Aggregation is going to be per time intervals ('h')
        importantCols = ['request_date']

        missingPerc = np.around(len(df[df[importantCols].isnull().any(axis=1)])/len(df)*100, decimals=3)
        if missingPerc<rowNanThreshold:
            logger.info("\tWe have {m}% missing values".format(m=missingPerc))
        else:
            logger.info("\tWe have more than {th}% missing values, attempting imputation".format(th=rowNanThreshold))
            df = self.impute(df)
        return df

    def load_weather_data(self):
        """
        Loads the Hourly sampled dataset with temperature, wind and rain information
        Returns:
            a dataframe containing weather data
        """
        logger.info("\tLoading additional weather data...")
        weatherdf = pd.read_csv(self.extfilepath)

        weatherdf = weatherdf.dropna(subset = ['datetime'])        # drop rows without a datetime column           
        weatherdf[self.weathercols] = weatherdf[self.weathercols].fillna(weatherdf[self.weathercols].median()) # for features, impute with median value

        return weatherdf

    def merge_weather_data(self, df):
        """
        Merges weather data to the original flight bookings data
        Args:
            df: a pandas dataframe of the original flight bookings data
        Returns:
            a pandas dataframe containing the joined data
        """
        #Load and resample df to a frequency of an hour
        dfw = self.load_weather_data()
        df["request_date"]= pd.to_datetime(df["request_date"])
        df = df.resample('h', on = 'request_date').count()              # read frequency from config
        df = df.rename(columns={'passenger_id': 'requests'})
        df = df[['requests']]

        # Reset Indexes to merge
        df = df.reset_index()
        dfw = dfw.reset_index()

        dfw['datetime'] = pd.to_datetime(dfw['datetime'])
        df = df.merge(dfw, left_on='request_date', right_on='datetime', how='left')

        #df = df.fillna(df.mean())
        df[self.weathercols] = df[self.weathercols].fillna(df[self.weathercols].median())
        df.drop(columns=['datetime', 'index'], inplace=True)                      # remove 2nd index

        # Reindex
        df = df.set_index('request_date')
        df = df.asfreq(freq='h')
        return df
