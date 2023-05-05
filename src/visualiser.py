from pyspark.sql import DataFrame
import pyspark.pandas as ps
import pandas as pd
from utils.get_spark import spark

from utils.logging import *
_logger = get_logger()


class DataVisualiser:

    def __init__(self, df: DataFrame):
        self.df = df

    def plot(self):
        pd.options.plotting.backend = "plotly"
        display(
            self.df.select("date", "measurement").pandas_api().set_index(["date"]).plot()
        )