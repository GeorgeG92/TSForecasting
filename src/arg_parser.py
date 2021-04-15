import os
import sys
import yaml
import argparse
import logging


def arg_parser(argv):
	"""This function implements an argument parser
	Args:
		argv: Ιnput arguments from the command line.
	Returns:
		An args object that contains parsed arguments.
	"""
	parser = argparse.ArgumentParser(description='Flight Bookings Forecasting')

	parser.add_argument('-l', '--logging', dest='logging_level', type=str.upper,
						default='info', help='set logging level',
						choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

	parser.add_argument('-m', '--method', default='SARIMAX', dest='method', type=str.upper,
						help='Set forecasting method', choices=['LSTM', 'SARIMAX'])

	parser.add_argument('-t','--train', action='store_true', default=False,       
					 help='Retrain model or load from disk')

	parser.add_argument('-g','--gridsearch', action='store_true', default=False,       
					 help='Retrain using GridSearchCV or use best found user-defined parameters')

	parser.add_argument('-c','--cluster', action='store_true', default=False,
					 help='Cluster the data on Lima map')

	parser.add_argument('-e','--explore', action='store_true', default=False,   
						dest='explore',
						help='Explore K-means Clustering and use the elbow rule to define optimal K')

	parser.add_argument('-i', '--impute', action='store_true', default=False,
						help='Select imputation fast (KNN) or slow (MICE) over dropping')

	parser.add_argument('--inputemethod', type=str.upper, choices=['KNN', 'MICE'],
						default='KNN', help='Imputation method choice')

	parser.add_argument('-d', '--datapath', dest='datapath',
						default=os.path.join('..','data'),
						help='The path to the data directory')
	
	parser.add_argument('-f', '--filename', dest='filename',
						default='routes.csv',
						help='The name of the data file')

	parser.add_argument('--modelpath', dest='modelpath',
						default=os.path.join('..', 'models'),
						help='The path to the model directory')

	parser.add_argument('--modelnamesarimax', dest='modelnamesarimax',
						default='SARIMAX_best_params.p',
						help='The name of the SARIMAX model')

	parser.add_argument('--modelnamelstm', dest='modelnamelstm',
						default='LSTM_best_params.h5',
						help='The name of the LSTM model')

	parser.add_argument('--configpath', dest='configpath',
						default=os.path.join('..','config'),
						help='The path to the config directory')
	
	parser.add_argument('--configname', dest='configname',
						default='config.yaml',
						help='The name of the config file')

	parser.add_argument('--configoptname', dest='configoptname',
						default='optimal_parameters.yaml',
						help='The name of the second config file')

	parser.add_argument('--exportpath', dest='exportpath',
						default=os.path.join('..','output'),
						help='The path to the output directory')

	parser.add_argument('--weatherfile', dest='weatherfile',
						default='lima_2015_weatherdata.csv',
						help='The name of the config file')

	args = parser.parse_args(argv[1:]) 
	logging.basicConfig(level=getattr(logging, args.logging_level), 
		format="%(asctime)s|%(filename)-20.20s||%(funcName)-20.20s|%(levelname)-8.8s|%(message)s")
		#format="%(asctime)s|%(filename)-20.20s|%(levelname)-8.8s|%(message)s")
		#format='%(asctime)s | %(levelname)8s | %(filename)s:%(funcName)30s | %(message)s')
	logger = logging.getLogger(__file__)
											  # exclude filename
	if args.train:
		if args.method=='LSTM' and not os.path.exists(args.modelpath):
			logger.warning("Train is set to False but model doesn't exist at {p}".format(p=os.path.join(rgs.modelpath, 'SARIMAX_best_params.p')))
			logger.warning("Setting train to True")
			args.train=True
		else:
			logger.warning("Train is set to False but model doesn't exist at {p}".format(p=os.path.join(args.modelpath, 'SARIMAX_best_params.p')))
			logger.warning("Setting train to True")
			#assert os.path.exists(args.modelpath, 'SARIMAX_best_params.p'), "Train is set to False but model doesn't exist at {p}".format(p=os.path.exists(args.modelpath, 'SARIMAX_best_params.p'))
			args.train = True

	if args.explore and not args.cluster:
		logger.error("Argument error: --explore argument requires --cluster")
		sys.exit(1)
	if not os.path.exists(os.path.join(args.modelpath, args.method)):
		logger.info("Creating Model path")
		os.makedirs(os.path.join(args.modelpath, args.method))
	args.modelpath = os.path.join(args.modelpath, args.method)
	if not os.path.exists(args.exportpath):
		logger.info("Creating exportpath")
		os.makedirs(os.path.join(args.exportpath))
		os.mkdir(os.path.join(args.exportpath, 'Cleaning'))
		os.mkdir(os.path.join(args.exportpath, 'Clustering'))
		os.mkdir(os.path.join(args.exportpath, 'TS_Decomposition'))
	assert os.path.exists(os.path.join(args.datapath, args.filename)), "Data file {file} does not exist".format(file=os.path.join(args.datapath, args.filename))
	if not os.path.exists(os.path.join(args.configpath, args.configoptname)):                # 
		args.useoptimalparams = False
	else:
		args.useoptimalparams = True
	return args