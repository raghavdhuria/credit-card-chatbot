import pandas as pd
import ast
import os
from typing import List, Dict, Any

class DataProcessor:
    """
    Class for processing credit and debit card data from XLSX or CSV file.
    Expected headers:
    Card Name, Issuer, Description, Annual Fee, Joining Fee,
    Category, Rewards Category, Eligibility
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        if not self.file_path or not os.path.exists(self.file_path):
            print("No data file provided or file doesn't exist. Creating empty DataFrame.")
            self.df = pd.DataFrame(columns=[
                'Card Name', 'Issuer', 'Description', 'Annual Fee', 'Joining Fee',
                'Category', 'Rewards Category', 'Eligibility'
            ])
            return self.df
            
        if self.file_path.lower().endswith('.csv'):
            self.df = pd.read_csv(self.file_path)
        else:
            self.df = pd.read_excel(self.file_path)
        print(f"Loaded {len(self.df)} cards")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        self.df = self.df.fillna('')

        # Convert numeric columns, handle ₹ sign if present
        for col in ['Annual Fee', 'Joining Fee']:
            self.df[col] = self.df[col].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(float)

        # Convert list-like columns from strings to lists
        for col in ['Rewards Category', 'Eligibility', 'Category']:
            self.df[col] = self.df[col].apply(self._parse_list_field)

        print("Data cleaning completed")
        return self.df

    def _parse_list_field(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
                return [str(parsed).strip()]
            except (ValueError, SyntaxError):
                # comma-separated fallback
                return [x.strip() for x in value.split(',') if x.strip()]
        return []

    def create_documents(self) -> List[Dict[str, Any]]:
        if self.df is None or self.df.empty:
            self.clean_data()

        documents = []
        for _, row in self.df.iterrows():
            text = f"""
Card Name: {row['Card Name']}
Issuer: {row['Issuer']}
Annual Fee: ₹{row['Annual Fee']}
Joining Fee: ₹{row['Joining Fee']}
Category: {', '.join(row['Category']) if row['Category'] else ''}
Rewards: {', '.join(row['Rewards Category']) if row['Rewards Category'] else ''}
Eligibility: {', '.join(row['Eligibility']) if row['Eligibility'] else ''}
Description: {row['Description']}
"""
            document = {
                "text": text.strip(),
                "metadata": {
                    "card_name": row["Card Name"],
                    "issuer": row["Issuer"],
                    "annual_fee": float(row["Annual Fee"]),
                    "joining_fee": float(row["Joining Fee"]),
                    "category": row["Category"],
                    "rewards_category": row["Rewards Category"],
                    "eligibility": row["Eligibility"],
                    "description": row["Description"],
                }
            }
            documents.append(document)

        self.processed_data = documents
        print(f"Created {len(documents)} documents for vector store")
        return documents
