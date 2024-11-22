import yfinance as yf
import pandas as pd


def load_data(ticker, start_date, end_date):
    """
    Load historical stock market data using yfinance.

    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :param start_date: Start date for the data (e.g., "2020-01-01").
    :param end_date: End date for the data (e.g., "2023-01-01").
    :return: DataFrame with historical stock data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker {
                         ticker} in the given date range.")
    return data


def preprocess_data(data):
    """
    Preprocess stock market data.

    :param data: DataFrame containing raw stock market data.
    :return: DataFrame with necessary columns and processed values.
    """
    # Keep only relevant columns
    processed_data = data[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Handle missing values
    processed_data = processed_data.dropna()

    # Reset index for easier manipulation
    processed_data.reset_index(inplace=True)

    print("Data preprocessing complete.")
    return processed_data


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    processed_data = preprocess_data(raw_data)

    print(processed_data.head())