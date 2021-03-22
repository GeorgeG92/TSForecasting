import yaml
import os
import logging

logger = logging.getLogger(__file__)

def config_parser(args):
	"""Open configuration file and parse it.
	Returns:
		A dictionary with validated parsed parameters.
	"""
	def assertBetween(x, lo, hi):
		if not (lo <= x <= hi):
			return False
		return True
	assert os.path.exists(os.path.join(args.configpath, args.configname)), "No configuration file found"
	with open(os.path.join(args.configpath, args.configname)) as file:
		config = yaml.load(file, Loader=yaml.FullLoader)
		assert isinstance(config['General']['cleanColumnsThresold'], float), "Colmun Threshold has to be a floating point number"
		assert assertBetween(config['General']['cleanColumnsThresold'], 0, 1), "Colmun Threshold has to be between 0 and 1"
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
	
	# Parse second config file
	if os.path.exists(os.path.join(args.configpath, args.configoptname)):
		with open(os.path.join(args.configpath, args.configoptname)) as file:
			configOpt = yaml.load(file, Loader=yaml.FullLoader)
			if args.method=='LSTM':
				if not 'Optimal LSTM Parameters' in configOpt:
					logger.warning("Mode is set to use optimal parameters for LSTM training but {conf} is malformed".format(conf=args.configoptname))
					args.useoptimalparams = False
				else:
					if not all (k in configOpt['Optimal LSTM Parameters'] for k in ("L2", "batchSize", 'dropout', 'epochs', 'learningRate', 'optimizer')): 
						logger.warning("Mode is set to use optimal parameters for LSTM training but {conf} is malformed".format(conf=args.configoptname))
						args.useoptimalparams = False
					else:
						config['Optimal LSTM Parameters'] = configOpt['Optimal LSTM Parameters']
			else:
				if not 'Optimal SARIMAX Parameters' in configOpt:
					logger.warning("Mode is set to use optimal parameters for SARIMAX training but {conf} is malformed".format(conf=args.configoptname))
					args.useoptimalparams = False
				else:
					if not all (k in configOpt['Optimal SARIMAX Parameters'] for k in ("p", "d", 'q', 'P', 'D', 'Q', 's')): 
						logger.warning("Mode is set to use optimal parameters for SARIMAX training but {conf} is malformed".format(conf=args.configoptname))
						args.useoptimalparams = False
					else:
						config['Optimal SARIMAX Parameters'] = configOpt['Optimal SARIMAX Parameters']
	return config
