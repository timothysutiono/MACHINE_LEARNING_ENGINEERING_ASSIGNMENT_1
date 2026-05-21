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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
import utils.data_processing_gold_feature_engineering


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_directory = "datamart/bronze/"

if not os.path.exists(bronze_directory):
    os.makedirs(bronze_directory)

#source systems to ingest
sources = {
    "loan_daily": "data/data/lms_loan_daily.csv",
    "attributes": "data/data/features_attributes.csv",
    "financials":   "data/data/features_financials.csv",
    "clickstream":  "data/data/feature_clickstream.csv",
}

# run bronze backfill
for date_str in dates_str_lst:
    for name, path in sources.items():
        utils.data_processing_bronze_table.process_bronze_table(date_str, name, path, bronze_directory, spark)

# create silver datalake
silver_directory = "datamart/silver/"

if not os.path.exists(silver_directory):
    os.makedirs(silver_directory)

# run silver backfill
for date_str in dates_str_lst:
    for source_name in sources:
        utils.data_processing_silver_table.process_silver_table(date_str, source_name, bronze_directory, silver_directory, spark)

#raise SystemExit("Silver done - stopping before silver/gold") #to be deleted

#Gold Feature Engineering
gold_fe_directory = "datamart/gold/feature_engineered/"
os.makedirs(gold_fe_directory, exist_ok=True)

fe = utils.data_processing_gold_feature_engineering
for date_str in dates_str_lst:
    fe.feature_engineer_loan_daily(date_str, silver_directory, gold_fe_directory, spark)
    fe.feature_engineer_attributes(date_str, silver_directory, gold_fe_directory, spark)
    fe.feature_engineer_financials(date_str, silver_directory, gold_fe_directory, spark)
    fe.feature_engineer_clickstream(date_str, silver_directory, gold_fe_directory, spark)

# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

#feature store
gold_feature_store_directory = "datamart/gold/feature_store/"
os.makedirs(gold_feature_store_directory, exist_ok=True)

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_feature_store(date_str, gold_fe_directory, gold_feature_store_directory,spark)

#Training table 
gold_training_directory = "datamart/gold/training/"
os.makedirs(gold_training_directory, exist_ok=True)

label_offset = 6
for date_str in dates_str_lst:
    snap = datetime.strptime(date_str, "%Y-%m-%d")
    if snap + relativedelta(months=-label_offset) < datetime.strptime(start_date_str, "%Y-%m-%d"):
        continue
    utils.data_processing_gold_table.process_training_table(date_str, gold_feature_store_directory, gold_label_store_directory, gold_training_directory, spark, label_offset_months=label_offset)

folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()



    