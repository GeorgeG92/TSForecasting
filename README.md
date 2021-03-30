# TSForecasting
Implementation of an end to end data pipeline from ETL, exploration, clustering, preprocessing, to time series forecasting using SARIMAX/LSTM models, built on top of statsmodels and tensorflow.keras.
Includes parallelized GridSearch, GPU support for LSTM mode and multiple Time Series Visualizations as part of the exploration process.

# Run Instructions:
Extract the data from data/routes.rar and run:
a) Using docker-compose
```sh
docker-compose up (--build)
```
b) install requirements through
```sh
pip install -r requirements.txt
cd src
python run.py
```


# Notes:
a) User has the option to edit the optimal_parameters.yaml file to broaden/shorten the range of hyperparameters that will be evaluated during the cross validation process.
b) SARIMAX GridSearch CV can take a long time to execute when a wide set of parameters to explore is set. Current optimal parameter range has been defined based on the analysis performed in notebooks/SARIMAX.ipynb notebook. 
c) In case of execution on a CUDA-enabled machine, to enable GPU acceleration for LSTM training, uncomment line 9 in docker-compose.yml (runtime: nvidia)
