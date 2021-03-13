# TSForecasting
Implementation of an end to end data pipeline from ETL, exploration, preprocessing, to time series forecasting using SARIMAX/LSTM models, built on top of statsmodels and tensorflow.keras.
Includes parallelized GridSearch, GPU support for LSTM mode and multiple Time Series Visualizations as part of the exploration process.

# Run Options:
a) Using docker-compose
```sh
docker-compose up (--build)
```
b) install requirements through
```sh
pip install -r requirements.txt
```
and run through
```sh
cd src
python run.py
```


