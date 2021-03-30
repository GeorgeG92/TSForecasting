import numpy as np
import warnings
import itertools
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from joblib import Parallel
from joblib import delayed
import logging

logger = logging.getLogger(__file__)


class SARIMAXModel():
	""" Implements the SARIMAX model class 
	"""
	def __init__(self, args, config, df):
		self.trainmodel = args.train
		self.stepsOut = config['Forecasting']['stepsOut']
		self.exploreParams = config['Forecasting']['SARIMAX']['GridSearchCV']
		self.exportpath = os.path.join(args.exportpath, 'SARIMAX')
		self.modelpath = os.path.join(args.modelpath, 'SARIMAX_best_params.p')
		self.useoptim = args.useoptimalparams
		if self.useoptim and 'Optimal SARIMAX Parameters' in config:
			self.bestparams = config['Optimal SARIMAX Parameters']
			self.optimExists = True
		else:
			self.optimExists = False
		rcParams['figure.figsize'] = 12, 6
		rcParams.update({'figure.autolayout': True})
		matplotlib.rc('font', size=14)
		matplotlib.rc('axes', titlesize=22)
		plt.gcf().autofmt_xdate()

		self.df = df
		self.df.index = pd.date_range(df.index[0], df.index[-1], freq='H')


	def generate_plots(self, df):
		""" Generates the ACF/PACF plots for the timeseries
		Args:
			df: the dataframe containing the time series data
		"""
		logger.info("\tExporting ACF/PACF plots...")
		params = {}
		params['s'] = 24     # seasonality is daily ~ 24h
		data =  df['requests']

		lag_acf = acf(data, nlags=50)
		lag_pacf = pacf(data, nlags=50)

		# Plot ACF:
		plt.figure(figsize=(15,7))
		plt.subplot(121)
		plt.stem(lag_acf)
		plt.axhline(y=0, linestyle='-', color='black')
		plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
		plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
		plt.xlabel('Lag')
		plt.ylabel('ACF')

		# Plot PACF:
		plt.subplot(122)
		plt.stem(lag_pacf)
		plt.axhline(y=0, linestyle='-', color='black')
		plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
		plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
		plt.xlabel('Lag')
		plt.ylabel('PACF')
		plt.savefig(os.path.join(self.exportpath, 'ACF_PACF_Plot.jpeg'))

	def score_sarimax(self, train, exog_train, params):
		""" Trains a SARIMAX model using given parameters
		Args:
			train: the data to train to
			exog_train: exogenous features for the time series
			params: a dictionary of parameters to train the model with
		Returns:
			A tuple of the parameters and the AIC score of the model fit
		"""
		self.model = SARIMAX(train, order=params[:3], seasonal_order=params[3:], freq='h', exog=exog_train)
		self.results = self.model.fit(disp=False, start_ar_lags=13)
		return (params, self.results.aic)

	def grid_search_sarimax(self):
		""" Performs GridSearch on SARIMAX training to determine the best model hyperaparameters
		Args:
			train: the data to train on
			exog_train: exogenous features besides the main time series feature
		"""
		params = self.exploreParams
		p = params['p']#range(0, 4)
		d = params['d']#range(0, 2)
		q = params['q']#range(0, 4)
		P = params['P']#range(0, 4)
		D = params['D']#range(0, 2)
		Q = params['Q']#range(0, 4)
		s = params['s']#range(24, 25)                                      
		pdq = list(itertools.product(p, d, q, P, D, Q, s))
		scores = None
		# Full parallel Hyperparameter GridSearchCV
		logger.info("\tStarting GridSearch for best hyperparameters, evaluating {num} models".format(num=len(pdq)))
		logger.info("\t\tEstimated Time: {num} mminutes".format(num=len(pdq)*0.4))
		executor = Parallel(n_jobs=-1, backend='threading')
		tasks = (delayed(self.score_sarimax)(self.train, self.exog_train, cfg) for cfg in pdq)
		scores = executor(tasks)
		# remove empty results due to non-convergeance
		scores = [r for r in scores if r[1] != None]
		scores.sort(key=lambda tup: tup[1])
		logger.info("\tLowest score is {score}, using parameters {params}".format(score=scores[0][1], params=scores[0][0]))
		return scores[0][0]

	def train_test_split(self):
		""" Splits both original and exogenous data into train/test slits
		Args:
			df: the dataframe with all the features (incl. exogenous)
			data: the flight bookings timeseries data
		"""
		self.train = self.data[:-(self.stepsOut)]
		self.test = self.data[-(self.stepsOut):]
		exog = self.df[['Temperature', 'Precipitation', 'WindSpeed']]
		self.exog_train = exog[:-(self.stepsOut)]
		self.exog_test = exog[-(self.stepsOut):]
		#return train, test, exog_train, exog_test

	def train_sarimax(self):
		""" 
		Trains a (series of) SARIMAX model(s) on the train data based on given configuration
		"""

		# Log Transform to battle Heteroscedasticity
		self.data = np.log(self.df['requests'])
		# Train/Test Split
		#train, test, exog_train, exog_test = self.train_test_split(self.df, data)
		self.train_test_split()

		if not self.trainmodel:
			logger.info("\tLoading Model from Disk...")
			self.result = SARIMAXResults.load(self.modelpath)
		elif self.useoptim:
			logger.info("\tTraining using optimal parameters")
			p,d,q,P,D,Q,s = self.bestparams['p'], self.bestparams['d'], self.bestparams['q'], self.bestparams['P'], self.bestparams['D'], self.bestparams['Q'], self.bestparams['s']
			self.model = SARIMAX(self.train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=self.exog_train)
			self.result = self.model.fit(disp=False)
			if not os.path.exists(self.modelpath):
				result.save(self.modelpath)
		else:
			logger.info("\tGridSearchCV SARIMAX to detect optimal parameters")
			start = time.time()
			bp = self.grid_search_sarimax()
			end = time.time()
			logger.info("\tGridSearchCV finished in {time} minutes".format(time=round((end-start)/60, 3)))

			# Retrain SARIMAX using best found hyperparameters
			p,d,q,P,D,Q,s = bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6]
			self.model = SARIMAX(self.train, order=(p,d,q),seasonal_order=(P,D,Q,s), exog=self.exog_train)
			self.result = self.model.fit(disp=False)
			self.result.save(self.modelpath)

			#Save optimal parameters to opt config
			best_params_dict = {'p':p,'d':d, 'q':q, 'P':P, 'D':D, 'Q':Q, 's':s}
			with open(os.path.join(args.configpath, args.configoptname)) as outfile:
				configOpt = yaml.load(file, Loader=yaml.FullLoader)
				configOpt['Optimal SARIMAX Parameters'] = best_params_dict
				yaml.dump(configOpt, outfile, default_flow_style=False)
		

	def predict_sarimax(self):
		""" 
		Runs model inference on the test data
		"""
		
		# Predict on the test data
		self.trainpreds = self.result.fittedvalues
		self.testpreds = self.result.predict(start=self.test.index[0], end=self.test.index[-1], exog=self.exog_test)
		ci = self.result.get_prediction(start=self.data.index[len(self.train)], end=self.data.index[len(self.data)-1] , exog=self.exog_test).conf_int()
		self.low = ci['lower requests']
		self.up = ci['upper requests']

		# Inverse scale the log transform
		self.dataRescaled = np.exp(self.data)
		self.trainRescaled = np.exp(self.train)
		self.testRescaled = np.exp(self.test)
		self.trainpredsRescaled = np.exp(self.trainpreds)
		self.testpredsRescaled = np.exp(self.testpreds)
		self.lowRestored = np.exp(self.low)
		self.upRestored = np.exp(self.up)

		# Evaluate
		score = mean_absolute_error(self.trainRescaled, self.trainpredsRescaled)
		logger.info("\tTrain MAE is {mae}".format(mae=score))
		score = mean_absolute_error(self.testRescaled, self.testpredsRescaled)
		logger.info("\tTest MAE is {mae}".format(mae=score))

		# Plot results and export
		self.plot_results()

	def plot_results(self):
		"""
		Generates forecast plots and exports them
		"""
		# Plot
		logger.info("\tExporting Forecast and Residual plots... ")
		plt.figure(figsize=(15,7))
		plt.subplot(211)
		plt.plot(self.dataRescaled, color='deepskyblue')
		plt.plot(self.trainpredsRescaled, color='navy')
		plt.plot(self.testpredsRescaled, color='lightcoral')
		plt.title('TS')
		plt.axvline(x=self.test.index[0], label='train/test split', c='mediumvioletred')
		plt.subplot(212)
		plt.plot(self.testpredsRescaled, color='lightcoral')
		plt.plot(self.testRescaled, color='deepskyblue')
		plt.fill_between(self.test.index, self.lowRestored, self.upRestored, color='pink')
		plt.title('Forecast')
		plt.savefig(os.path.join(self.exportpath, 'SARIMAX_Forecast.png'))
		plt.close()

		residualPlot = self.result.plot_diagnostics(figsize = (15, 8), lags=20)
		residualPlot.savefig(os.path.join(self.exportpath, 'residualPlot.png'))

	def model_sarimax(self):
		""" Responsible for the train/inference pipeline of the model
		Args:
			df: the dataframe containing the time series data
		"""
		logger.info("SARIMAX Modeling")
		self.train_sarimax()
		self.predict_sarimax()
