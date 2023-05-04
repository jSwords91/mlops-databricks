# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # MLOps End-to-End
# MAGIC
# MAGIC This notebook will soon be split in to numerous .py files. However whilst I am writing the code I will keep it in a single notebook.

# COMMAND ----------

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as ps

from utils.logging import *
_logger = get_logger()


# COMMAND ----------

# DBTITLE 1,Data Loader
FILEPATH = "dbfs:/FileStore/shared_uploads/jswords@aiimi.com/daily_minimum_temperatures_in_me.csv"

@dataclass
class DataLoaderConfig:
    """
     Attributes:
        file_type (str): File type of data being read in
        file_path (str): Filep path of data being read in
    """
    file_type: str = "csv"
    file_path: str = FILEPATH
    date_column: str = "date"
    value_column: str = "Daily minimum temperatures"


class DataLoader:
    """
    Class for reading in a file and cleaning the data
    """

    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg

    @staticmethod
    def read_data(file_path: str, file_type: Optional[str] = "csv") -> DataFrame:
        return (
            spark.read.format(file_type)
            .option("header", "true")
            .load(file_path)
        )

    def rename_columns(self, df: DataFrame) -> DataFrame:
        return (df
                .withColumnRenamed(self.cfg.value_column, "measurement")
                .withColumnRenamed(self.cfg.date_column, "date")
                )

    @staticmethod
    def change_column_type(df: DataFrame) -> DataFrame:
        return (df
                .withColumn("measurement", col("measurement").cast(DoubleType()))
                .withColumn("date", to_date(col("date"), "M/d/yyyy"))
                )

    @staticmethod
    def set_index_and_sort(df: DataFrame) -> DataFrame:
        return df.sort("date")

    def load_and_clean_pyspark(self) -> DataFrame:

        _logger.info('Reading Data & Applying Pre-Processing Steps')
        df = self.read_data(self.cfg.file_path, self.cfg.file_type)
        df_renamed = self.rename_columns(df)
        df_typed = self.change_column_type(df_renamed)
        df_cleaned = self.set_index_and_sort(df_typed)
        return df_cleaned

 

# COMMAND ----------

display(DataLoader(DataLoaderConfig).load_and_clean_pyspark())

# COMMAND ----------

# DBTITLE 1,Visualise

class DataVisualiser:

    def __init__(self, df: DataFrame):
        self.df = df

    def plot(self):
        pd.options.plotting.backend = "plotly"
        display(
            self.df.select("date", "measurement").pandas_api().set_index(["date"]).plot()
        )

# COMMAND ----------

DataVisualiser(df).plot()

# COMMAND ----------

# DBTITLE 1,Feature Engineering
@dataclass
class FeaturizerConfig:
    date_col_name: str = "date"

class Featurizer:

    def __init__(self, cfg: FeaturizerConfig):
        self.cfg = cfg

    @staticmethod
    def add_date_features(df: DataFrame, date_col_name: str):
        return (df
        .withColumn("day_of_year", dayofyear(date_col_name))
        .withColumn("quarter", quarter(date_col_name))
        .withColumn("month", month(date_col_name))
        .withColumn("year", year(date_col_name))
        )

    def run(self, df: DataFrame):
        _logger.info('Adding Date Features')
        return self.add_date_features(df, self.cfg.date_col_name)


# COMMAND ----------

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



# COMMAND ----------

import databricks.automl
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
 

@dataclass
class AutoMLConfig:
    target_col: str = "measurement"
    date_col: str = "date"
    forecast_horizon: int = 30
    forecast_frequency: str = "d"
    primary_metric: str = "rmse"
    output_database: str = "default"

class RunAutoML:
    
    def __init__(self, cfg: AutoMLConfig):
        self.cfg = cfg

    def run(self, df: DataFrame):
        return (
            databricks.automl.forecast(
            df,
            target_col=self.cfg.target_col,
            time_col=self.cfg.date_col,
            horizon=self.cfg.forecast_horizon,
            frequency=self.cfg.forecast_frequency,
            primary_metric=self.cfg.primary_metric,
            output_database=self.cfg.output_database)
        )



# COMMAND ----------

summary = RunAutoML(AutoMLConfig).run(df)


run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id
 
model_uri = "runs:/{run_id}/model".format(run_id=trial_id)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
forecasts = pyfunc_model._model_impl.python_model.predict_timeseries(include_history=True)

# COMMAND ----------

forecast_pd = spark.table(summary.output_table_name)
display(forecast_pd)
