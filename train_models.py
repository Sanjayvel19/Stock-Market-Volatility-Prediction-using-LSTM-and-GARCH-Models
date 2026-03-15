import os
import sys
import joblib
import pandas as pd

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.data_loader import load_data
from models.GARCH_Model import run_garch_all
from models.lstm_model import run_lstm_with_returns
from models.panel_Data import run_panel_analysis

def train_and_save():
    # Removed emojis to prevent UnicodeEncodeError
    print("--- STARTING OFFLINE TRAINING PHASE ---")
    
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        print("Created 'saved_models' directory.")

    try:
        # 1. Panel Analysis
        print("\n[1/3] Running Panel Data Analysis...")
        _, panel_summary, _ = run_panel_analysis()
        joblib.dump(panel_summary, 'saved_models/panel_results.pkl')
        print("SUCCESS: Panel Analysis Saved.")

        # 2. GARCH Volatility
        print("\n[2/3] Estimating GARCH Volatility...")
        garch_results = run_garch_all()
        joblib.dump(garch_results, 'saved_models/garch_results.pkl')
        print("SUCCESS: GARCH Forecasts Saved.")

        # 3. LSTM Predictions
        print("\n[3/3] Training LSTM Models for all 10 stocks...")
        lstm_results = run_lstm_with_returns()
        lstm_df = pd.DataFrame(lstm_results)
        joblib.dump(lstm_df, 'saved_models/lstm_results.pkl')
        print("SUCCESS: LSTM Predictions Saved.")

        print("\n========================================")
        print("TRAINING COMPLETE: 10 Stocks Processed.")
        print("Location: /saved_models/")
        print("========================================")

    except Exception as e:
        print(f"ERROR during training: {e}")

if __name__ == "__main__":
    train_and_save()