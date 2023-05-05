# Databricks notebook source
import os
import datetime as dt
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as md

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
mlflow.autolog(disable=True)

from statsmodels.tsa.api import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

from pyspark.sql.functions import *
from pyspark.sql.types import *

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date
from pyspark.sql import DataFrame
import pyspark.pandas as psp

from src.data_loader import *
from src.featurizer import *
from src.visualiser import *
from src.auto_ml import *

from utils.logging import *
_logger = get_logger()

# COMMAND ----------

df = Featurizer(FeaturizerConfig).run(
    DataLoader(DataLoaderConfig).load_and_clean_pyspark()
).toPandas()

# COMMAND ----------

FORECAST_HORIZON = 30

is_history = [True] * (len(df) - FORECAST_HORIZON) + [False] * FORECAST_HORIZON
train = df.iloc[is_history]
test = df.iloc[~np.array(is_history)]


# COMMAND ----------

train

# COMMAND ----------

from prophet import Prophet
import logging

# COMMAND ----------

# set model parameters
model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )
 
# fit the model to historical data
model.fit(train.rename(columns={"date" : "ds", "measurement": "y"}))

# COMMAND ----------

# define a dataset including both historical dates & 90-days beyond the last available date
future_pd = model.make_future_dataframe(
  periods=FORECAST_HORIZON, 
  freq='d', 
  include_history=True
  )
 
# predict over the dataset
forecast_pd = model.predict(future_pd)
 
display(forecast_pd)

# COMMAND ----------

predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')
 
# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

# COMMAND ----------

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date
 

# calculate evaluation metrics
mae = mean_absolute_error(test, forecast_pd.iloc[-FORECAST_HORIZON:, :]['yhat'].values)
mse = mean_squared_error(test, forecast_pd.iloc[-FORECAST_HORIZON:, :]['yhat'].values)
rmse = sqrt(mse)
 
# print metrics to the screen
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------


