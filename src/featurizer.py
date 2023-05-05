from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as ps

from utils.get_spark import spark

from utils.logging import *
_logger = get_logger()


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
