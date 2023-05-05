# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # ML Experimentation
# MAGIC
# MAGIC This will not be an exhaustive attempt at modelling, but will show how you can use Hyper Opt in an attempt to beat the AutoML baseline.
# MAGIC
# MAGIC
# MAGIC **AutoML Baseline: 2.58~**

# COMMAND ----------

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as psp

import pandas as pd
import numpy as np

from src.data_loader import *
from src.featurizer import *
from src.visualiser import *
from src.auto_ml import *

from prophet import Prophet
import logging

import mlflow
import hyperopt
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope

mlflow.autolog(disable=True)


from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

from utils.logging import *
_logger = get_logger()

# COMMAND ----------

def make_train_test(df: pd.DataFrame, forecast_horizon: int) -> Union[pd.DataFrame, pd.DataFrame]: 
    is_history = [True] * (len(df) - forecast_horizon) + [False] * forecast_horizon
    train = df.iloc[is_history]
    test = df.iloc[~np.array(is_history)]
    return train, test


# COMMAND ----------

FORECAST_HORIZON = 30

# COMMAND ----------

df = (
    Featurizer(FeaturizerConfig)
    .run(DataLoader(DataLoaderConfig).load_and_clean_pyspark())
    .toPandas()
)

# COMMAND ----------

train, test = make_train_test(df, FORECAST_HORIZON)

# COMMAND ----------


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> float:
    mse = mean_squared_error(actual, predicted)
    return sqrt(mse)

def make_future_df(model: Prophet, periods: int, freq: str) -> pd.DataFrame:
    return model.make_future_dataframe(
        periods=periods, 
        freq=freq, 
        include_history=True
    )

# Define objective function
def objective(params):

    # set model parameters
    model = Prophet(
        interval_width=0.95,
        growth=params["growth"],
        daily_seasonality=params["daily_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        yearly_seasonality=params["yearly_seasonality"],
        seasonality_mode=params["seasonality_mode"],
    )

    # fit the model to historical data
    model.fit(train.rename(columns={"date": "ds", "measurement": "y"}))

    future_pd = make_future_df(model, FORECAST_HORIZON, "d")

    forecast_pd = model.predict(future_pd)

    # score on actual v pred on test set
    score = evaluate(
        test["measurement"].values,
        forecast_pd.iloc[-FORECAST_HORIZON:, :]["yhat"].values,
    )

    return score

# COMMAND ----------

# Define search space
search_space = {
    "seasonality_mode": hp.choice("seasonality_mode", ["multiplicative", "additive"]),
    "growth": hp.choice("growth", ["linear", "logistic", "flat"]),
    "yearly_seasonality": hp.choice("yearly_seasonality", [False, True, 6, 8, 10, 12]),
    "weekly_seasonality": hp.choice("weekly_seasonality", [False, True, 1, 3, 7]),
    "daily_seasonality": hp.choice("daily_seasonality", [False, True])
    }

# Set parallelism (should be order of magnitude smaller than max_evals)
spark_trials = SparkTrials(parallelism=3)

with mlflow.start_run(run_name="Hyperopt"):
    argmin = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=spark_trials,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Success
# MAGIC
# MAGIC We have successfully imprvoved upon AutoML. We've gone from **2.58~** to **2.51~**.
# MAGIC
# MAGIC We could do even better and make use of the engineered features we added, rather than simply our y value.

# COMMAND ----------

best_params = space_eval(search_space, argmin)
print(best_params)

# COMMAND ----------

model = Prophet(**best_params)

# fit the best model to historical data
model.fit(train.rename(columns={"date": "ds", "measurement": "y"}))

future_pd = make_future_df(model, FORECAST_HORIZON, "d")

forecast_pd = model.predict(future_pd)

# score on actual v pred on test set
score = evaluate(
    test["measurement"].values,
    forecast_pd.iloc[-FORECAST_HORIZON:, :]["yhat"].values,
)

print(score)

# COMMAND ----------

predict_fig = model.plot(forecast_pd, xlabel='', ylabel='Measurement')
 
xlim = predict_fig.axes[0].get_xlim()
new_xlim = (xlim[1]-(180.0+365.0), xlim[1]-120)
predict_fig.axes[0].set_xlim(new_xlim)

# COMMAND ----------


