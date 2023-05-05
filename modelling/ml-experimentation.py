# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # ML Experimentation
# MAGIC
# MAGIC This will not be an exhaustive attempt at modelling, but will show how you can use Hyper Opt in an attempt to beat the AutoML baseline.
# MAGIC
# MAGIC
# MAGIC **AutoML RMSE Baseline: 2.58~**

# COMMAND ----------

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

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
from utils.notebook_utils import * 
_logger = get_logger()

# COMMAND ----------

def make_train_test(
    df: pd.DataFrame, forecast_horizon: int
) -> Union[pd.DataFrame, pd.DataFrame]:
    is_history = [True] * (len(df) - forecast_horizon) + [False] * forecast_horizon
    train = df.iloc[is_history]
    test = df.iloc[~np.array(is_history)]
    return train, test

# COMMAND ----------

FORECAST_HORIZON = 30

# COMMAND ----------

config = create_data_loader_config("base_params")

df = (
    Featurizer(FeaturizerConfig)
    .run(DataLoader(config).run())
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

with mlflow.start_run(run_name="Hyperopt") as mlflow_run:
    argmin = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=spark_trials,
    )

# COMMAND ----------

mlflow.end_run()

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


with mlflow.start_run():
    model = Prophet(**best_params).fit(train.rename(columns={"date": "ds", "measurement": "y"}))

    future_pd = make_future_df(model, FORECAST_HORIZON, "d")
    forecast_pd = model.predict(future_pd)

    score = evaluate(
    test["measurement"].values,
    forecast_pd.iloc[-FORECAST_HORIZON:, :]["yhat"].values,
    )

    mlflow.prophet.log_model(model, artifact_path="prophet-tuned-model")
    mlflow.log_params(best_params)
    model_uri = mlflow.get_artifact_uri("prophet-tuned-model")
    print(f"Model artifact logged to: {model_uri}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Use new model to predict

# COMMAND ----------

loaded_model = mlflow.prophet.load_model(model_uri)

forecast = loaded_model.predict(loaded_model.make_future_dataframe(60))

print(f"forecast:\n${forecast.head(30)}")

# COMMAND ----------

predict_fig = model.plot(forecast_pd, xlabel='', ylabel='Measurement')
 
xlim = predict_fig.axes[0].get_xlim()
new_xlim = (xlim[1]-(180.0+365.0), xlim[1]-120)
predict_fig.axes[0].set_xlim(new_xlim)

# COMMAND ----------


