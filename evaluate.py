import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def evaluate_predictions(stock_data, predictions, scaler, seq_length):
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actual_prices = stock_data['Close'].values[seq_length:]
    rmse = calculate_rmse(actual_prices, predictions[:len(actual_prices)])
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    print("This file is not meant to be run directly. Please run main.py.")
