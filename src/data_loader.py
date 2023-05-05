from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

from utils.get_spark import spark
from utils.logging import *
from utils.notebook_utils import *

_logger = get_logger()


@dataclass
class DataLoaderConfig:
    """
     Attributes:
        file_type (str): File type of data being read in
        file_path (str): File path of data being read in
        date_column(str): Column containing the date
        value_column (str): Column containing the y value
    """
    file_path: str
    file_type: str 
    date_column: str
    value_column: str



def create_data_loader_config(config_path: str) -> DataLoaderConfig:
    pipeline_config = load_config(config_path)
    return DataLoaderConfig(
        file_path=pipeline_config["input_data_file_path"],
        file_type=pipeline_config["input_file_type"],
        date_column=pipeline_config["date_column"],
        value_column=pipeline_config["value_column"],
    )


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

    def run(self) -> DataFrame:
        """
        Run feature table creation pipeline
        """
        _logger.info('==========Data Ingest==========')
        _logger.info('Reading Data & Applying Pre-Processing Steps')
        df = self.read_data(self.cfg.file_path, self.cfg.file_type)
        df_renamed = self.rename_columns(df)
        df_typed = self.change_column_type(df_renamed)
        df_cleaned = self.set_index_and_sort(df_typed)
        return df_cleaned

 