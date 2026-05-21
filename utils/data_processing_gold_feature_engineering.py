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

#LOAN_DAILY 
def feature_engineer_loan_daily(snapshot_date_str, silver_directory, gold_fe_directory, spark):
    date_tag = snapshot_date_str.replace('-', '_')
    df = spark.read.parquet(f"{silver_directory}silver_loan_daily_{date_tag}.parquet")

    #repayment ratio: how much of the payment due has been paid by cust
    df = df.withColumn("repayment_ratio", F.when(col("due_amt") > 0, col("paid_amt") / col("due_amt")).otherwise(None))

    #Overdue signal 
    df = df.withColumn("is_overdue", F.when(col("overdue_amt") > 0, 1).otherwise(0).cast(IntegerType()))

    out = f"{gold_fe_directory}gold_fe_loan_daily_{date_tag}.parquet"
    df.write.mode("overwrite").parquet(out)
    print("saved to:", out)
    return df

#ATTRIBUTES
def feature_engineer_attributes(snapshot_date_str, silver_directory, gold_fe_directory, spark):
    date_tag = snapshot_date_str.replace('-', '_')
    df = spark.read.parquet(f"{silver_directory}silver_attributes_{date_tag}.parquet")

    #Age Grouping
    df = df.withColumn("age_group", F.when(col("Age") < 26, "18-25")
                    .when(col("Age") < 36, "26-35")
                    .when(col("Age") < 51, "36-50")
                    .when(col("Age").isNotNull(), "51+")
                    .otherwise(None))

    out = f"{gold_fe_directory}gold_fe_attributes_{date_tag}.parquet"
    df.write.mode("overwrite").parquet(out)
    print("saved to:", out)
    return df

#FINANCIALS
def feature_engineer_financials(snapshot_date_str, silver_directory, gold_fe_directory, spark):
    date_tag = snapshot_date_str.replace('-', '_')
    df = spark.read.parquet(f"{silver_directory}silver_financials_{date_tag}.parquet")

    #debt to income ratio (measuring credit risk)
    df = df.withColumn("debt_to_income", F.when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income")).otherwise(None))

    #EMI to income ratio (measuring how much of your monthly income goes to loan repayment)
    df = df.withColumn("emi_to_income", F.when(col("Monthly_Inhand_Salary") > 0, col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")).otherwise(None))

    #Make Credit Mix Ordinal
    df = df.withColumn("credit_mix_score", F.when(col("Credit_Mix") == "Good", 2)
                       .when(col("Credit_Mix") == "Standard", 1)
                       .when(col("Credit_Mix") == "Bad", 0)
                       .otherwise(None).cast(IntegerType()))
    
    #Likely higher risk Loans
    df = df.withColumn("has_payday_loan", F.when(col("Type_of_Loan").contains("Payday Loan"), 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("has_student_loan", F.when(col("Type_of_Loan").contains("Student Loan"), 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("has_mortgage", F.when(col("Type_of_Loan").contains("Mortgage Loan"), 1).otherwise(0).cast(IntegerType()))

    out = f"{gold_fe_directory}gold_fe_financials_{date_tag}.parquet"
    df.write.mode("overwrite").parquet(out)
    print("saved to:", out)
    return df

#CLICKSTREAM
def feature_engineer_clickstream(snapshot_date_str, silver_directory, gold_fe_directory, spark):
    date_tag = snapshot_date_str.replace('-', '_')
    df = spark.read.parquet(f"{silver_directory}silver_clickstream_{date_tag}.parquet")

    #clickstream total (sum of all clickstream features)
    df = df.withColumn("clickstream_total", col("fe_1") + col("fe_2") + col("fe_3") + col("fe_4")
                       + col("fe_5") + col("fe_6") + col("fe_7") + col("fe_8")
                       + col("fe_9") + col("fe_10") + col("fe_11") + col("fe_12")
                       + col("fe_13") + col("fe_14") + col("fe_15") + col("fe_16")
                       + col("fe_17") + col("fe_18") + col("fe_19") + col("fe_20"))

    out = f"{gold_fe_directory}gold_fe_clickstream_{date_tag}.parquet"
    df.write.mode("overwrite").parquet(out)
    print("saved to:", out)
    return df