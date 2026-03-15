import numpy as np
import pandas as pd
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# 1. Setup path to find 'utils' from the 'models' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_loader import load_data

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_sequences(data, lookback):
    """Converts time-series data into windows for LSTM training."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def run_lstm_with_returns():
    """
    Trains an LSTM model for all 10 stocks and returns the predicted 
    return for the next trading day.
    """
    # Synchronized Tickers with data_loader.py
    stocks = [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "KPITTECH.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "OFSS.NS"
    ]

    data = load_data()
    lookback = 500  # Reduced to  to ensure newer stocks have enough history
    stock_stats = []

    # Ensure the storage directory exists
    os.makedirs("saved_models", exist_ok=True)

    for stock in stocks:
        if stock not in data:
            print(f"⚠️ Warning: {stock} not found in data_loader. Skipping...")
            continue

        print(f"--- Training LSTM: {stock} ---")
        df = data[stock]
        
        # Use 'Close' price and handle any missing values
        prices = df["Close"].ffill().dropna().values.reshape(-1, 1)

        if len(prices) <= lookback:
            print(f"⚠️ Skipping {stock}: Only {len(prices)} days found. Need > {lookback}.")
            continue

        # 1. Scale Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        current_price = float(prices[-1])

        # 2. Prepare Sequences
        X, y = create_sequences(scaled_data, lookback)
        
        # Split into Train (80%) and Val (20%)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # 3. Build Model Architecture (Optimized for fast training)
        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # 4. Train with Early Stopping
        callback = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        model.fit(
            X_train, y_train, 
            batch_size=32, 
            epochs=10, # Kept at 10 for speed during your project demo
            validation_data=(X_val, y_val), 
            callbacks=[callback], 
            verbose=0
        )

        # 5. Predict the next day
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
        prediction_scaled = model.predict(last_sequence, verbose=0)
        predicted_price = float(scaler.inverse_transform(prediction_scaled)[0][0])

        # 6. Calculate Expected Return (Alpha)
        expected_return = (predicted_price - current_price) / current_price

        stock_stats.append({
            "Stock": stock,
            "Current_Price": round(current_price, 2),
            "Predicted_Price": round(predicted_price, 2),
            "Expected_Return": float(expected_return)
        })
        
        # Save model for later use in app.py
        model.save(f"saved_models/{stock.split('.')[0]}_lstm.h5")
        
    return stock_stats

if __name__ == "__main__":
    results = run_lstm_with_returns()
    df = pd.DataFrame(results)
    
    # Sort by Expected Return
    df = df.sort_values(by="Expected_Return", ascending=False)
    print("\n--- LSTM Return Predictions (All 10 Stocks) ---")
    print(df)