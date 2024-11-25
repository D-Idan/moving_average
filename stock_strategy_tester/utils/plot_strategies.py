import matplotlib.pyplot as plt

from data.data_loader import load_data, preprocess_data


def plot_moving_averages(data, short_window, long_window):
    """
    Plot the benchmark price and the moving averages on the same plot.

    :param data: DataFrame containing stock data with a 'Close' column.
    :param short_window: Lookback period for the short moving average.
    :param long_window: Lookback period for the long moving average.
    """
    if "Close" not in data.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    # Calculate short and long moving averages
    data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data["Close"], label="Benchmark (Close Price)", color="blue")
    plt.plot(data.index, data["SMA_Short"], label=f"Short Moving Average ({short_window} days)", color="red")
    plt.plot(data.index, data["SMA_Long"], label=f"Long Moving Average ({long_window} days)", color="green")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Benchmark Price and Moving Averages")
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    import yfinance as yf

    # Load sample data
    ticker = "AAPL"
    # ticker = "TSLA"
    start_date = "2017-01-01"
    end_date = "2024-01-01"
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Plot the benchmark price and moving averages
    plot_moving_averages(data, short_window=60, long_window=112)