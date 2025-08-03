#!/usr/bin/env python3
"""
Credit Card Recommendation System - Notebook Runner
Extracted from CreditCard_Recommendation.ipynb
"""

# Setup and Imports
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Get configuration variables from environment
config = {
    "weaviate_cluster_url": os.getenv("WEAVIATE_CLUSTER_URL"),
    "weaviate_api_key": os.getenv("WEAVIATE_API_KEY"),
    "tavily_api_key": os.getenv("TAVILY_API_KEY"),
    "credit_card_data_path": os.getenv("CREDIT_CARD_DATA_PATH")
}

print("Loaded configuration:")
print(config)

# Load your credit/debit card dataset using the provided path
file_path = config["credit_card_data_path"]

# Read Excel or CSV file based on file extension
if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
    df = pd.read_excel(file_path)
else:
    df = pd.read_csv(file_path)

print(f"Dataset columns: {df.columns.to_list()}")

# Select only the relevant columns
columns_to_keep = [
    "Card Name",
    "Issuer",
    "Description",
    "Annual Fee",
    "Joining Fee",
    "Category",
    "Rewards Category",
    "Eligibility"
]

df_cleaned = df[columns_to_keep].copy()

# Function to parse list-like fields stored as strings
def parse_list_field(field):
    if isinstance(field, list):
        return field
    if isinstance(field, str):
        try:
            import ast
            parsed = ast.literal_eval(field)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
            return [str(parsed).strip()]
        except:
            return [x.strip() for x in field.split(',') if x.strip()]
    return []

# Apply parsing on list-like columns
for col in ["Rewards Category", "Eligibility", "Category"]:
    df_cleaned[col] = df_cleaned[col].apply(parse_list_field)

# Clean numeric columns: remove symbols and convert to floats
for col in ["Annual Fee", "Joining Fee"]:
    df_cleaned[col] = df_cleaned[col].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(float)

print(f"Number of credit/debit cards loaded: {len(df_cleaned)}")

# Show the first 10 rows
print("\nFirst 10 rows of cleaned data:")
print(df_cleaned.head(10))

# Optionally save cleaned dataset
cleaned_data_path = "Cleaned_Credit_Card_Data.xlsx"
df_cleaned.to_excel(cleaned_data_path, index=False)
print(f"\nCleaned dataset saved as {cleaned_data_path}")

print("\n✅ Credit Card data processing completed successfully!")