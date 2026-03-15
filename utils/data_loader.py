import yfinance as yf
import pandas as pd
import numpy as np

# 10 Indian IT stocks (including LTIM and others discussed)
STOCKS = [
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "KPITTECH.NS", "MPHASIS.NS", "COFORGE.NS", 
    "PERSISTENT.NS", "OFSS.NS"
]

def download_raw_data(tickers, start="2015-01-01", end="2025-12-31"):
    """Downloads adjusted close data from Yahoo Finance."""
    print(f"Downloading data for: {tickers}")
    data = yf.download(tickers, start=start, end=end, group_by="ticker")
    return data

def add_indicators(df):
    """Adds technical indicators to a individual stock DataFrame."""
    # Returns (Target variable for many models)
    df["Returns"] = df["Close"].pct_change()

    # Trend Indicators
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # Volatility
    df["Volatility"] = df["Returns"].rolling(window=20).std()

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    
    return df

def load_data():
    """
    Main function called by other scripts. 
    Returns a dictionary of DataFrames keyed by ticker.
    """
    raw = download_raw_data(STOCKS)
    dataset = {}

    for stock in STOCKS:
        try:
            # Extract OHLCV and copy to avoid SettingWithCopyWarning
            df = raw[stock][["Open", "High", "Low", "Close", "Volume"]].copy()
            
            # Apply indicators
            df = add_indicators(df)
            
            # Drop rows with NaN from rolling calculations
            df = df.dropna()
            
            dataset[stock] = df
        except KeyError:
            print(f"Warning: Data for {stock} not found.")
            
    return dataset

def get_panel_data():
    """
    Converts the dictionary into a single Multi-Index DataFrame.
    Essential for 'Panel Data.py' regressions.
    """
    data_dict = load_data()
    # Stack the dictionary into one DataFrame
    panel_df = pd.concat(data_dict.values(), keys=data_dict.keys())
    panel_df.index.names = ['Ticker', 'Date']
    return panel_df

if __name__ == "__main__":
    # Test the loader
    print("--- Testing Data Loader ---")
    data = load_data()
    if "TCS.NS" in data:
        print("\nTop 5 rows for TCS.NS with Indicators:")
        print(data["TCS.NS"].head())
    
    print("\n--- Testing Panel Format ---")
    panel = get_panel_data()
    print(panel.head())