from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as ps

import databricks.automl
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
 
from utils.get_spark import spark

from utils.logging import *
_logger = get_logger()


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
        _logger.info('Running AutoML')
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

