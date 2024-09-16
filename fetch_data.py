import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """_summary_

    Args:
        ticker (_type_): _description_
        start_date (_type_): _description_
        end_date (_type_): _description_
    """
    stock_data = yf.download(ticker, start_date, end_date)
    return stock_data