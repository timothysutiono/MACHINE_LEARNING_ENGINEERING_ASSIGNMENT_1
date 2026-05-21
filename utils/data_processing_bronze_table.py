import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, source_name, source_csv_path, bronze_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    # load data - IRL ingest from back end source system
    df = (spark.read.csv(source_csv_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date))
    print(f"{source_name} {snapshot_date_str} row count:", df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = bronze_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df