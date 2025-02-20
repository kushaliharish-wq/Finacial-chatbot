import re
import os
import pyodbc
from openai import OpenAI
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import datetime
import json
import mysql.connector
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm.auto import tqdm
import time
from dotenv import load_dotenv

# Step 1: Connect to Azure SQL Database
AZURE_SQL_CONNECTION_STRING = AZURE_SQL_CONNECTION_STRING

engine = create_engine(AZURE_SQL_CONNECTION_STRING)

# Load Data from CSV
transactions = pd.read_csv("/personal_transactions.csv")
budgets = pd.read_csv("/Budget.csv")

# Data Cleaning
transactions["Date"] = pd.to_datetime(transactions["Date"], format="%m/%d/%Y")
transactions["Category"] = transactions["Category"].str.strip().str.lower()
budgets["Category"] = budgets["Category"].str.strip().str.lower()

# Drop duplicates from transactions and budgets based on their respective columns
transactions.drop_duplicates(inplace=True)
budgets.drop_duplicates(subset="Category", inplace=True)  # Ensure distinct categories

# Print the columns of 'budgets' to check the exact name of the budget column
print("Columns in budgets DataFrame:", budgets.columns)

# Ensure 'budget' column exists and fill missing values
if "Budget" in budgets.columns:
    budgets["Budget"] = budgets["Budget"].fillna(0)  # Replace NaN budgets with 0
else:
    print("Error: 'Budget' column not found in 'budgets' DataFrame")

# Ensure all other columns have no missing values
budgets["Category"] = budgets["Category"].fillna(
    "misc"
)  # Replace missing categories with 'misc'
transactions["Category"] = transactions["Category"].fillna(
    "misc"
)  # Same for transactions

# Ensure all other metadata columns (like 'account_name', 'transaction_type') are filled
transactions["Account Name"] = transactions["Account Name"].fillna("unknown")
transactions["Transaction Type"] = transactions["Transaction Type"].fillna("unknown")

# Verify if there are any remaining NaN values
if budgets.isnull().any().any() or transactions.isnull().any().any():
    print("Warning: Some NaN values remain in the data after cleaning.")

print("\n✅ Data Cleaning Done!")

# Rename Columns
transactions_df = transactions.rename(
    columns={
        "Date": "date",
        "Description": "description",
        "Amount": "amount",
        "Category": "transaction_category",
        "Account Name": "account_name",
        "Transaction Type": "transaction_type",
    }
)

budgets_df = budgets.rename(columns={"Category": "category", "Budget": "budget"})

# Insert Data into Azure SQL
transactions_df.to_sql("Transactions", engine, if_exists="append", index=False)
budgets_df.to_sql("Budgets", engine, if_exists="append", index=False)

print("✅ Data successfully inserted into Azure SQL!")
