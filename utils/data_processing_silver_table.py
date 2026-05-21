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


def process_silver_table(snapshot_date_str, source_name, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_tag = snapshot_date_str.replace('-', '_')
    
    # connect to bronze table
    bronze_path = bronze_directory + "bronze_" + source_name + "_" + date_tag + ".parquet"
    df = spark.read.parquet(bronze_path)
    print('loaded from:', bronze_path, 'row count:', df.count())

    # clean data: enforce schema / data type
    if source_name == "loan_daily":
        df = clean_loan_daily(df)
    elif source_name == "attributes":
        df = clean_attributes(df)
    elif source_name == "financials":
        df = clean_financials(df)
    elif source_name == "clickstream":
        df = clean_clickstream(df)
    else:
        raise ValueError("Unknown source: " + source_name)
    
    # save silver table - IRL connect to database to write
    silver_path = silver_directory + "silver_" + source_name + "_" + date_tag + ".parquet"
    df.write.mode("overwrite").parquet(silver_path)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', silver_path)
    
    return df

#LOAN_ DAILY 
def clean_loan_daily(df):
    # clean data: enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(), 
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    #force casting new type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    #drop constants - 0 info
    df = df.drop("tenure", "loan_amt")

    #Whitespace deletion
    df = df.withColumn("Customer_ID", F.trim(col("Customer_ID")))

    return df

#ATTRIBUTES
def clean_attributes(df):
    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(), 
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    #clean data: Age outside 18-100 made null
    df = df.withColumn("Age", F.when((col("Age") >= 18) & (col("Age") <= 100), col("Age")).otherwise(None))

    #clean data: Occupation placeholder "_______" made null
    df = df.withColumn("Occupation", F.when(col("Occupation") == "_______", None).otherwise(col("Occupation")))

    #clean data: SSN gibberish
    df = df.withColumn("SSN", F.when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"),col("SSN")).otherwise(None))
    
    #Whitespace deletion
    df = df.withColumn("Customer_ID", F.trim(col("Customer_ID")))

    #drop PII SSN and Name 
    df = df.drop("SSN", "Name")

    return df


#FINANCIALS
def clean_financials(df):
    # clean_data: remove "_" from numeric columns before casting
    numeric_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    for c in numeric_cols:
        df = df.withColumn(c, F.regexp_replace(col(c).cast("string"), "_", ""))

    #Clean Data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    #clean data: Credit_Mix placeholder "_" to be null
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))

    #Bounding Values: Nullify the values that dont' make sense
    #Delay_from_due_date not bounded because negative numbers are customers paying early 
    for c, lo, hi in [ 
        ("Num_of_Loan", 0, 20),
        ("Num_Bank_Accounts", 0, 20),
        ("Num_Credit_Card", 0, 20),
        ("Num_of_Delayed_Payment", 0, 100),
        ("Interest_Rate", 0, 50),
        ("Num_Credit_Inquiries", 0, 50),
        ]:
        df = df.withColumn(c, F.when((col(c) >= lo) & (col(c) <= hi), col(c)).otherwise(None))

    #Null Values in type_of_loans means no loan not missing data 
    df = df.withColumn("Type_of_Loan", F.when(col("Type_of_Loan").isNull(), "No Loan").otherwise(col("Type_of_Loan")))

    #Nullify NM in Payment_of_Min_Amount
    df = df.withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount") == "NM", None).otherwise(col("Payment_of_Min_Amount")))

    #Nullify garbage symbols in Payment_Behaviour
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour").rlike(r"[!@#%]"), None).otherwise(col("Payment_Behaviour")))

    #Whitespace deletion
    df = df.withColumn("Customer_ID", F.trim(col("Customer_ID")))

    #Anomaly in Monthly_Balance made null
    df = df.withColumn("Monthly_Balance", F.when(col("Monthly_Balance") >= -100000, col("Monthly_Balance")).otherwise(None))
    
    #10,000 of Amount_Invested_Monthly seems too round and looks like anomalies 
    df = df.withColumn("Amount_invested_monthly", F.when(col("Amount_invested_monthly") == 10000, None).otherwise(col("Amount_invested_monthly")))
    
    return df

#CLICKSTREAM
def clean_clickstream(df):
    #clean data: enforce schema / data type
    column_type_map = {}
    for i in range(1,21):
        column_type_map["fe_" + str(i)] = IntegerType()
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    #Whitespace deletion
    df = df.withColumn("Customer_ID", F.trim(col("Customer_ID")))

    return df 