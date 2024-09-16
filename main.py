import pandas as pd
import numpy as np
from fetch_data import fetch_stock_data
from preprocess_data import preprocess_data
from predict import format_data_for_gpt4, generate_predictions
from evaluate import evaluate_predictions

# Parameters
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-01-01"
seq_length = 60
api_key = 'YOUR_GPT4_API_KEY'

# Step 1: Fetch stock data
stock_data = fetch_stock_data(ticker, start_date, end_date)
stock_data.to_csv("stock_data.csv")

# Step 2: Preprocess data
sequences, scaler = preprocess_data(stock_data, seq_length)
np.save("sequences.npy", sequences)
np.save("scaler.npy", scaler)

# Step 3: Generate predictions
formatted_data = format_data_for_gpt4(sequences)
predictions = generate_predictions(formatted_data, api_key)
np.save("predictions.npy", predictions)

# Step 4: Evaluate predictions
evaluate_predictions(stock_data, predictions, scaler, seq_length)
