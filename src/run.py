from data_loader import DataLoader
from arg_parser import arg_parser
from config_parser import config_parser
from cluster_analysis import ClusterAnalysis
from TS_analysis import TS_Analysis
from LSTM_model import LSTMModel
from SARIMAX_model import SARIMAXModel
import os
import sys
import yaml
import warnings
import logging

def main(args, config):
    data, df = DataLoader(args).process_data()
    if args.cluster:
        clusterAnalysis(args, df)
    TS_Analysis(args, data)
    if args.method=='LSTM':
        LSTMModel(args, config, data)
    elif args.method=='SARIMAX':
        SARIMAXModel(args, config, data).model_sarimax()                  
    else:
        SARIMAXModel(args, config, data).model_sarimax() 
        LSTMModel(args, config, data)
    logging.info("Done")



if __name__== "__main__":
    args = arg_parser(sys.argv)
    config = config_parser(args)
    logging.basicConfig(level=getattr(logging, args.logging_level), 
        format='%(asctime)s | %(levelname)-8s | %(filename)s:%(funcName)-30s | %(message)s')
    main(args, config)
