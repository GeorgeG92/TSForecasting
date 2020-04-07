from dataLoader import DataLoader
from clusterAnalysis import clusterAnalysis
from TS_Analysis import TS_Analysis
from LSTMModel import LSTMModel
import os
import sys
import yaml
import argparse
import warnings
import pandas as pd

def main(args, config, cluster=False):
    data = DataLoader(args.input, args.impute).getData()
    if args.cluster:
        clusterAnalysis(data, args, explore=args.explore)
    TS_Analysis(data)
    #data = pd.read_csv("data.csv")                       # 2 be removed
    if args.method=='LSTM':
        LSTMModel(data, args.train, config)
    elif args.method=='SARIMAX':
        SARIMAXModel(data, config)                       # implement SARIMAX logic
    else:
        SARIMAXModel(data, config)
        LSTMModel(data, args.train, config)
    return 0

def argParser(argv):
    """Add command line arguments and parse user inputs.
    Args:
        argv: User input from the command line.
    Returns:
        An args object that contains parsed arguments.
    """
    # Creating a parser
    parser = argparse.ArgumentParser(description='Flight Bookings Forecasting')
    parser.add_argument('-t','--train', action='store_true', default=False,       # for booleans: store_true
                     help='Retrain model with GridSearchCV')

    parser.add_argument('-c','--cluster', action='store_true', default=False,
                     help='Cluster the data on Lima map')

    parser.add_argument('-e','--explore', action='store_true', default=False,     # for strings:
                         help='Explore K-means Clustering and use the elbow rule to define optimal K')

    parser.add_argument('-d', '--data', dest='input',
                        default='../data/routes.csv',
                        help='Override path to data file')

    parser.add_argument('-l', '--logging', dest='logging_level',
                        default='info', help='set logging level',
                        choices=['debug', 'info', 'warning', 'error',
                                 'critical'])
    parser.add_argument('-i', '--impute', action='store_true', default=False,
                        help='Select imputation fast (KNN) or slow (MICE) over dropping')

    parser.add_argument('-m', '--method',
                        default='LSTM',
                        help='Set forecasting method',
                        choices=['LSTM',
                                 'SARIMAX'])

    args = parser.parse_args(argv[1:])                                           # exclude filename
    if args.explore and not args.cluster:
        print("Argument error: --explore argument requires --cluster")
        sys.exit(1)
    return args


def ConfigParser():
    """Open configuration file and parse it.
    Returns:
        A dictionary with validated parsed parameters.
    """
    def assertBetween(x, lo, hi):
        if not (lo <= x <= hi):
            return False
        return True

    assert os.path.exists("./config.yaml"), "No configuration file found"
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        assert os.path.exists(config['General']['input']), "Input file not found"
        if not os.path.exists(config['General']['enrichment']):
            config['General']['enrichment'] = None
        assert isinstance(config['General']['cleanColumnsThresold'], float), "Colmun Threshold has to be a floating point number"
        assert assertBetween(config['General']['cleanColumnsThresold'], 0, 1), "Colmun Threshold has to be between 0 and 1"
        assert config['General']['imputation'] in ['KNN', 'MICE'], "Imputation methods supported are 'KNN' and 'MICE'"
        assert config['General']['resample'] in ['h'], "Invalid time sampling frequency"
        if config['General']['resample']!='h':
            config['General']['enrichment'] = None
        assert isinstance(config['Forecasting']['stepsOut'], int), "StepsOut parameter has to be an integer"
        assert isinstance(config['Forecasting']['stepsIn'], int), "StepsIn parameter has to be an integer"
        assert all(isinstance(item, int)  for item in config['Forecasting']['LSTM']['GridSearchCV']['epochs']), "Epochs parameter has to be an integer number"
        assert all(isinstance(item, float) | isinstance(item, int)  for item in config['Forecasting']['LSTM']['GridSearchCV']['L2']), "L2 Regularization parameter has to be a float number"
        assert all(assertBetween(element, 0, 1) for element in config['Forecasting']['LSTM']['GridSearchCV']['L2']), "L2 has to be between 0 and 1"
        assert all(isinstance(item, int)  for item in config['Forecasting']['LSTM']['GridSearchCV']['batchSize']), "BatchSize parameter has to an integer number"
        assert all(isinstance(item, int) | isinstance(item, float) for item in config['Forecasting']['LSTM']['GridSearchCV']['dropout']), "Dropout parameter has to be a float number"
        assert all(assertBetween(element, 0, 1) for element in config['Forecasting']['LSTM']['GridSearchCV']['dropout']), "Dropout has to be between 0 and 1"
        assert all(isinstance(item, float) for item in config['Forecasting']['LSTM']['GridSearchCV']['learningRate']), "LearningRate parameter has to be a float number"
        assert all(assertBetween(element, 0, 1) for element in config['Forecasting']['LSTM']['GridSearchCV']['learningRate']), "LearningRate has to be between 0 and 1"
        # assert isinstance(config['Forecasting']['testSize'], float), "TestSize has to be a floating point number"
        # assert assertBetween(config['Forecasting']['testSize'], 0, 1), "TestSize has to be between 0 and 1"
        return config


if __name__== "__main__":
    warnings.filterwarnings("ignore")
    args = argParser(sys.argv)
    config = ConfigParser()
    main(args, config)
