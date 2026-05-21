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


def process_labels_gold_table(snapshot_date_str, silver_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

#FEATURE STORE
def process_feature_store(snapshot_date_str, gold_fe_directory, gold_feature_store_directory, spark):
    date_tag = snapshot_date_str.replace('-', '_')

    loan = spark.read.parquet(f"{gold_fe_directory}gold_fe_loan_daily_{date_tag}.parquet")
    attr = spark.read.parquet(f"{gold_fe_directory}gold_fe_attributes_{date_tag}.parquet")
    fin = spark.read.parquet(f"{gold_fe_directory}gold_fe_financials_{date_tag}.parquet")
    clk = spark.read.parquet(f"{gold_fe_directory}gold_fe_clickstream_{date_tag}.parquet")

    df = (loan
          .join(attr, on=["Customer_ID", "snapshot_date"], how="left")
          .join(fin, on=["Customer_ID", "snapshot_date"], how="left")
          .join(clk, on=["Customer_ID", "snapshot_date"], how="left"))
    
    out = f"{gold_feature_store_directory}gold_feature_store_{date_tag}.parquet"
    df.write.mode("overwrite").parquet(out)
    print("saved to:", out)
    return df

#TRAINING TABLE
def process_training_table(label_date_str, gold_feature_store_directory, gold_label_store_directory, gold_training_directory, spark, label_offset_months=6):
    label_tag = label_date_str.replace('-', '_')
    snap = datetime.strptime(label_date_str, "%Y-%m-%d")
    feature_tag = (snap + relativedelta(months=-label_offset_months)).strftime('%Y_%m_%d')

    feature_path = f"{gold_feature_store_directory}gold_feature_store_{feature_tag}.parquet"
    label_path = f"{gold_label_store_directory}gold_label_store_{label_tag}.parquet"

    if not os.path.exists(feature_path):
        print(f"skip {label_date_str}: no feature_store at {feature_tag}")
        return None
    
    features = spark.read.parquet(feature_path)
    labels = spark.read.parquet(label_path)

    #join feature and label & drop 1 snapshot date and 1 cust ID
    df = labels.join(features.drop("snapshot_date", "Customer_ID"),
                     on = "loan_id", how="inner")
    
    out = f"{gold_training_directory}gold_training_{label_tag}.parquet"
    df.write.mode("overwrite").parquet(out)
    print(f"saved to: {out} rows = {df.count()}")
    return df 

