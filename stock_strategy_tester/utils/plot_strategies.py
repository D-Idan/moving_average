import matplotlib.pyplot as plt

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
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    data = yf.download(ticker, start=start_date, end=end_date)

    # Plot the benchmark price and moving averages
    plot_moving_averages(data, short_window=20, long_window=50)