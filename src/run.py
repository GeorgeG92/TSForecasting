from dataLoader import DataLoader
from clusterAnalysis import clusterAnalysis
from TS_Analysis import TS_Analysis
from forecastModel import forecastModel
import os
import sys
import argparse
import warnings


def main(args, cluster=False):
    data = DataLoader(args.input, args.impute).getData()
    # if args.cluster:
    #     clusterAnalysis(data, args, explore=args.explore)
    # TS_Analysis(data)
    # model = forecastModel(data)
    return 0

def arg_parser(argv):
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
    parser.add_argument('-i', '--impute',
                        default='fast',
                        help='Set imputation method: fast (KNN) or slow (MICE)',
                        choices=['fast',
                                 'slow'])

    parser.add_argument('-m', '--method',
                        default='LSTM',
                        help='Set forecasting method',
                        choices=['LSTM',
                                 'SARIMAX'])

    args = parser.parse_args(argv[1:])                                           # exclude filename

    assert os.path.exists(args.input), "Data file is missing"

    if args.explore and not args.cluster:
        print("Argument error: --explore argument requires --cluster")
        sys.exit(1)
    return args


if __name__== "__main__":
    # os.environ['KMP_WARNINGS'] = 'off'        # openmp warnings
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow deprications
    # warnings.filterwarnings("ignore")
    args = arg_parser(sys.argv)
    print(args)
    main(args)
