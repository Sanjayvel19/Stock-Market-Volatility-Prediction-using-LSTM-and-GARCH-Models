import os
import sys
import pandas as pd
import numpy as np
from arch import arch_model 

# 1. Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Standard Import
from utils.data_loader import load_data

def run_garch_all():
    """Fits GARCH(1,1) using the last 500 trading days."""
    data_dict = load_data()
    results = {}

    for stock, df in data_dict.items():
        try:
            # Check if 'Returns' exists
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()
            
            # 500-DAY FIX: Take only the most recent 500 rows
            if len(df) > 500:
                df_subset = df.tail(500).copy()
            else:
                df_subset = df.copy()

            print(f"--- Estimating GARCH for: {stock} (Using {len(df_subset)} days) ---")
            
            # Scale returns to 100 for better GARCH convergence
            returns = df_subset['Returns'].dropna() * 100

            if len(returns) < 50:
                print(f"Skipping {stock}: Not enough data.")
                continue

            # Fit GARCH(1,1)
            model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
            fitted = model.fit(disp="off")
            
            # Forecast 1-step ahead
            forecast = fitted.forecast(horizon=1)
            next_day_variance = forecast.variance.iloc[-1].values[0]
            
            # Save as a list for the app.py merging logic
            results[stock] = [float(next_day_variance)]

        except Exception as e:
            print(f"Error modeling {stock}: {e}")
            continue

    return results