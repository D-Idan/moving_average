import pandas as pd


def moving_average_strategy(data, short_window=20, long_window=50):
    """
    Generate buy/sell signals based on a simple moving average (SMA) crossover strategy.

    :param data: DataFrame containing stock data with a 'Close' column.
    :param short_window: Lookback period for the short moving average.
    :param long_window: Lookback period for the long moving average.
    :return: Series of signals: 1 for buy, -1 for sell, 0 for hold.
    """
    if "Close" not in data.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    # Calculate short and long moving averages
    data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()

    # Generate signals: 1 for buy, -1 for sell, 0 for hold
    data["Signal"] = 0
    data.loc[data["SMA_Short"] > data["SMA_Long"], "Signal"] = 1  # Buy
    data.loc[data["SMA_Short"] < data["SMA_Long"], "Signal"] = -1  # Sell

    return data["Signal"]


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Load sample data
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    data = yf.download(ticker, start=start_date, end=end_date)

    # Apply moving average strategy
    signals = moving_average_strategy(data)
    data["Signal"] = signals

    print(data[["Close", "SMA_Short", "SMA_Long", "Signal"]].tail())
