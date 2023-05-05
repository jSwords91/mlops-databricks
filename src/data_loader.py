from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

from utils.get_spark import spark
from utils.logging import *

_logger = get_logger()

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
        return (spark.read.format(file_type)
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

 