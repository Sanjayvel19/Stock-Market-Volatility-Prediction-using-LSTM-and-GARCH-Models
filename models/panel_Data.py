import os
import sys
import pandas as pd

# 1. Path Fix: This tells Python to look in the 'MBA PROJECT' folder for imports
current_dir = os.path.dirname(os.path.abspath(__file__)) # gets 'models' folder
project_root = os.path.dirname(current_dir)             # gets 'MBA PROJECT' root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Now the import will work perfectly
from utils.data_loader import load_data

def run_panel_analysis():
    """
    Performs Panel Data Exploration and Feature Engineering 
    without econometric regression.
    """
    # 1. Load dataset (Expected format: {ticker: dataframe})
    data_dict = load_data()

    # 2. Convert dictionary to a Long-Format Panel Dataframe
    # This creates a Multi-Index (Stock, Date) structure
    df_panel = pd.concat(data_dict.values(), keys=data_dict.keys())
    df_panel.index.names = ['Stock', 'Date']
    
    # 3. Feature Engineering: Cross-sectional Comparisons
    # Calculate returns for each stock independently within the panel
    df_panel['Daily_Return'] = df_panel.groupby('Stock')['Close'].pct_change()
    
    # Calculate 20-day Rolling Volatility (Historical)
    df_panel['Hist_Volatility'] = df_panel.groupby('Stock')['Daily_Return'].transform(
        lambda x: x.rolling(window=20).std()
    )

    # 4. Global Summary Statistics (Aggregated by Stock)
    # This identifies high-performing vs high-risk stocks over the whole period
    panel_summary = df_panel.groupby('Stock').agg({
        'Daily_Return': ['mean', 'std', 'skew', 'kurt'],
        'Close': ['min', 'max', 'last'],
        'RSI': 'mean',
        'Volume': 'mean'
    })

    # 5. Cross-Sectional Correlation Matrix
    # Shows how stocks move in relation to each other
    correlation_matrix = df_panel.reset_index().pivot(
        index='Date', columns='Stock', values='Daily_Return'
    ).corr()

    return df_panel, panel_summary, correlation_matrix

if __name__ == "__main__":
    print("Executing Panel Data Exploration...")

    panel_df, summary, correlations = run_panel_analysis()

    print("\n--- PANEL DATA STRUCTURE (First 5 Rows) ---")
    print(panel_df[['Close', 'Daily_Return', 'Hist_Volatility']].dropna().head())

    print("\n--- DESCRIPTIVE PANEL SUMMARY ---")
    # This helps in identifying which stocks are outliers
    print(summary)

    print("\n--- INTER-STOCK CORRELATION (Returns) ---")
    # Useful for portfolio diversification logic
    print(correlations.round(2))