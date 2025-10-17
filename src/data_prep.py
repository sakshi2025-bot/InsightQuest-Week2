# data_prep.py

import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Loads the dataset and converts it to a DataFrame."""
    try:
        # Using 'latin1' encoding as noted in your report
        df = pd.read_csv(file_path, encoding='latin1')
        print(f"✅ Data loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}. Check your 'data/' folder.")
        return None

def handle_missing_values(df):
    """
    Handles missing values by applying median imputation for numerical data
    and mode imputation for categorical/object data.
    """
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Use median for numerical columns
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Use mode for categorical/object columns
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("✅ Missing values handled successfully.")
    return df

def preprocess_data(df):
    """Fixes date formats and adds the 'Profit Margin' calculated field."""
    
    # 1. Fix date/time formats
    # Your report mentions 'Order Date' and 'Ship Date'
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    print("✅ Date columns formatted successfully.")

    # 2. Add calculated field: "Profit Margin (%)"
    # Ensure no division by zero using np.where
    df['Profit Margin (%)'] = np.where(
        df['Sales'] != 0,
        (df['Profit'] / df['Sales']) * 100,
        0
    )
    print("✅ Calculated field 'Profit Margin (%)' added.")
    
    return df

# --- Main Execution Block ---

if __name__ == "__main__":
    # Define file paths relative to the root project folder
    INPUT_FILE_PATH = os.path.join("data", "sales.csv") 
    OUTPUT_FILE_PATH = os.path.join("data", "cleaned_sales_data.csv")
    
    print("--- Starting Week 1 Data Preprocessing Pipeline ---")

    # 1. Load Data
    raw_df = load_data(INPUT_FILE_PATH)
    if raw_df is None:
        exit() # Stop if data loading failed

    # 2. Handle Missing Values
    df_cleaned = handle_missing_values(raw_df.copy())

    # 3. Preprocess (Dates and Calculated Fields)
    df_processed = preprocess_data(df_cleaned)

    # 4. Save the Final Output for Week 2
    df_processed.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\n✨ SUCCESS! Cleaned data saved to: {OUTPUT_FILE_PATH}")

    # Display summary statistics as requested in your original report
    print("\n--- Summary Statistics of Key Metrics ---")
    print(df_processed[['Sales', 'Profit', 'Profit Margin (%)']].describe())