# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # MLOps End-to-End
# MAGIC
# MAGIC This notebook dmeonstrates the process that our pipelines will follow by reading in the .py files that contain our classes. 
# MAGIC
# MAGIC In practice you wouldn't use a notebook (although you could).
# MAGIC
# MAGIC Broadly, the steps I will follow are:
# MAGIC
# MAGIC * Load the data 
# MAGIC * Apply data cleaning methods
# MAGIC * Feature Engineering
# MAGIC * Baseline model using AutoML [This step wouldn't be included in the pipeline]
# MAGIC   * Experiment to see if we can beat the AutoML model [This step wouldn't be included in the pipeline]
# MAGIC * Save the best model for use in the pipeline
# MAGIC * Apply the best model to data
# MAGIC * Monitor model performance

# COMMAND ----------

# DBTITLE 1,Libraries
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as ps

from src.data_loader import *
from src.featurizer import *
from src.visualiser import *
from src.auto_ml import *

from utils.logging import *
_logger = get_logger()


# COMMAND ----------

# DBTITLE 1,Data Loader
display(DataLoader(DataLoaderConfig).load_and_clean_pyspark())

# COMMAND ----------

# DBTITLE 1,Visualise
DataVisualiser(df).plot()

# COMMAND ----------

# DBTITLE 1,Feature Engineering
df = Featurizer(FeaturizerConfig).run(
    DataLoader(DataLoaderConfig).load_and_clean_pyspark()
)

# COMMAND ----------

display(df)

# COMMAND ----------

@dataclass
class MLflowTrackingConfig:
    """
    Configuration data class used to unpack MLflow parameters during a model training run.
    Attributes:
        run_name (str)
            Name of MLflow run
        experiment_id (int)
            ID of the MLflow experiment to be activated. If an experiment with this ID does not exist, raise an exception.
        experiment_path (str)
            Case sensitive name of the experiment to be activated. If an experiment with this name does not exist,
            a new experiment wth this name is created.
        model_name (str)
            Name of the registered model under which to create a new model version. If a registered model with the given
            name does not exist, it will be created automatically.
    """
    run_name: str
    experiment_id: int = None
    experiment_path: str = None
    model_name: str = None

# COMMAND ----------

# DBTITLE 1,AutoML: Baseline
summary = RunAutoML(AutoMLConfig).run(df)

# COMMAND ----------


run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id
 
model_uri = "runs:/{run_id}/model".format(run_id=trial_id)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
forecasts = pyfunc_model._model_impl.python_model.predict_timeseries(include_history=True)

forecast_pd = spark.table(summary.output_table_name)
display(forecast_pd)

# COMMAND ----------

display(df)

# COMMAND ----------


