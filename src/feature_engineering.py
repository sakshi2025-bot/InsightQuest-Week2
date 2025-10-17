import pandas as pd
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose 
import warnings
warnings.filterwarnings('ignore') 

# --- NEW IMPORTS FOR VISUALIZATION ---
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------------

# --- FIX: Set non-interactive backend for saving images in terminal environments ---
# This fixes silent failures during plotting in non-GUI environments (like MINGW64).
plt.switch_backend('Agg') 

# Define file paths (GLOBAL SCOPE)
CLEANED_DATA_PATH = os.path.join("data", "cleaned_sales_data.csv")
PREPARED_DATA_PATH = os.path.join("data", "sales_prepared.csv") 

# --- GLOBAL DEFINITION OF PLOT PATH (Fixes UndefinedVariableError) ---
PLOTS_DIR = os.path.join("reports", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True) # Ensure the directory exists
# ----------------------------------------------------------------------


def run_week2_feature_engineering():
    """Performs all required Week 2 tasks (EDA & Feature Engineering) and saves the final prepared dataset."""
    
    print("\n--- Starting Week 2: Deep EDA & Feature Engineering ---")
    
    # 1. Load the cleaned data from Week 1 (Task 5)
    try:
        df = pd.read_csv(
            CLEANED_DATA_PATH,
            encoding='utf-8',           
            parse_dates=['Order Date'],  
            on_bad_lines='skip',
            index_col=False
        )
    except Exception as e:
        print(f"FATAL ERROR: Failed to read CSV file. Check file integrity or encoding. Error: {e}")
        return
    
    # --- Data Preparation & Indexing ---
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
    df['Profit Margin (%)'] = (df['Profit'] / df['Sales']) * 100 
    
    # CRITICAL FIX: Ensure 'Order Date' is set as the index
    df.set_index('Order Date', inplace=True)
    
    # --- Task 5: Feature Engineering for Forecasting ---
    print("✅ Adding Time-Based Features (Month, Quarter, Rolling Sales)...") 
    
    # 5a. Aggregate daily data to monthly frequency 
    sales_ts_series = df.groupby(df.index.to_period('M'))['Sales'].sum().to_timestamp(how='start').rename('Sales')
    sales_ts = sales_ts_series.to_frame() 

    # 5b. Add day-level features
    df['Year'] = df.index.year 
    df['Month'] = df.index.month 
    df['Quarter'] = df.index.quarter 
    df['Days_in_Month'] = df.index.days_in_month 

    # 5c. Calculate Lagging/Rolling Features (on monthly sales_ts)
    sales_ts['Rolling_3M_Sales'] = sales_ts['Sales'].rolling(window=3, min_periods=1).mean() 
    sales_ts['Rolling_6M_Sales'] = sales_ts['Sales'].rolling(window=6, min_periods=1).mean() 
    sales_ts['Sales_Last_Year'] = sales_ts['Sales'].shift(12)
    df['YOY_Change'] = ((df['Sales'] - df['Sales'].shift(12)) / df['Sales'].shift(12)) * 100
    # --- Fill NaN values for modeling readiness and cleaner visualization ---
# Fills all NaN values in the YOY_Change column with 0.
    df['YOY_Change'] = df['YOY_Change'].fillna(0)
    
    # NOTE: Re-calculating YOY_Change on the daily DF for completeness, though monthly is often preferred for forecasting.
    sales_ts['YOY_Change'] = ((sales_ts['Sales'] - sales_ts['Sales_Last_Year']) / sales_ts['Sales_Last_Year']) * 100 
    
    # 5d. Merge the monthly features back to the daily DataFrame
    sales_ts_reset = sales_ts.reset_index().rename(columns={'Order Date': 'Month_Key'}) 
    df['Month_Key'] = df.index.to_period('M').to_timestamp(how='start')
    
    df = df.merge(sales_ts_reset[['Month_Key', 'Rolling_3M_Sales', 'Rolling_6M_Sales', 'YOY_Change']], 
                  on='Month_Key', 
                  how='left',
                  suffixes=('_daily', '_monthly'))
    
    df.drop(columns=['Month_Key'], inplace=True)
    
    # --- Task 1: Advanced Time Series Analysis (Calculations) ---
    print("✅ Performing Time Series Decomposition & Volatility Analysis...")
    
    # 1a. Decompose total sales 
    try:
        decomposition = seasonal_decompose(sales_ts['Sales'].dropna(), model='additive', period=12)
        print("\t-> Sales decomposed into Trend, Seasonality, and Residuals.")
        # 1b. Peak/Slump Identification is done by analyzing the decomposition and monthly totals.
    except Exception as e:
        print(f"\t-> Decomposition failed (Check data length/frequency): {e}")

    # 1c. Calculate sales volatility (std dev) per region/month and merge
    sales_volatility = df.groupby(['Region', 'Month'])['Sales'].std(ddof=0).reset_index().rename(
        columns={'Sales': 'Sales_Volatility_Monthly'}
    )
    df = df.merge(sales_volatility, on=['Region', 'Month'], how='left')
    
    # --- Task 1 & 2 VISUALIZATION ---
    print("✅ Generating required line chart and hierarchical product plot...")

    # 1. Plot Monthly Sales Trends (Line Chart - Satisfies plotting requirement)
    plt.figure(figsize=(12, 6))
    plt.plot(sales_ts.index, sales_ts['Sales'], label='Monthly Sales', alpha=0.6)
    plt.plot(sales_ts.index, sales_ts['Rolling_3M_Sales'], label='3M Rolling Average', color='red', linewidth=2.5)
    plt.title('Monthly Sales Trend with Rolling Average')
    plt.xlabel('Date')
    plt.ylabel('Total Sales ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_sales_trend.png'))
    plt.close()
    print("\t-> Monthly Sales Trend chart saved.")


    # 2. Hierarchical Product Performance (Stacked Bar Chart - Alternative to Treemap)
    product_hierarchy = df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    product_pivot = product_hierarchy.pivot_table(index='Region', columns='Category', values='Sales', fill_value=0)

    plt.figure(figsize=(10, 7))
    product_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
    
    plt.title('Hierarchical Sales Performance by Region and Category')
    plt.xlabel('Region')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=0)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'hierarchical_product_performance.png'))
    plt.close()
    print("\t-> Hierarchical Product Performance chart saved.")
    
    # --- Task 2 & 3: Product Analysis & Region-Category Interactions ---
    print("✅ Calculating Metrics, Identifying Top Products, and Analyzing Interactions...")
    
    # 2b & 3a: Compute Revenue per Product per Region 
    product_revenue = df.groupby(['Region', 'Product Name'])['Sales'].transform('sum')
    df['Revenue_Per_Product_Region'] = product_revenue
    
    # 2a. Identify Top 10 profit-making and Top 10 low-margin products (Output to console)
    product_performance = df.groupby('Product Name').agg(
        Total_Profit=('Profit', 'sum'),
        Avg_Profit_Margin=('Profit Margin (%)', 'mean')
    ).sort_values(by='Total_Profit', ascending=False)

    print("\n[EDA Insight] Top 10 Profit-Making Products:\n", product_performance.head(10))
    print("\n[EDA Insight] Top 10 Low-Margin Products (by Avg Margin):\n", product_performance.sort_values(by='Avg_Profit_Margin', ascending=True).head(10))

    # 3b. Build pivot tables (Output to console)
    try:
        pivot_revenue = df.pivot_table(index='Region', columns='Category', values='Sales', aggfunc='sum', fill_value=0)
        print("\n[EDA Insight] Total Revenue by Region x Category:\n", pivot_revenue)
        
        pivot_profit_margin = df.pivot_table(index='Category', values='Profit Margin (%)', aggfunc='mean', fill_value=0)
        print("\n[EDA Insight] Average Profit Margin by Category:\n", pivot_profit_margin.sort_values(by='Profit Margin (%)', ascending=False))
    except KeyError as e:
        print(f"\n[Warning] Pivot tables require the column: {e}. Ensure it exists in cleaned data.")

    # 3c. Create a correlation heatmap of numerical features
    numerical_features = ['Sales', 'Discount', 'Quantity', 'Profit', 'Profit Margin (%)']
    correlation_matrix = df[numerical_features].corr()
    print("\n[EDA Insight] Correlation Matrix:\n", correlation_matrix)
    
    # --- Final Save (Task 5) ---
    df.to_csv(PREPARED_DATA_PATH, index=True) 
    print(f"\n✨ SUCCESS! Week 2 tasks completed. Data saved to: {PREPARED_DATA_PATH}")
    print(f"✨ All plots saved to the '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    run_week2_feature_engineering()
