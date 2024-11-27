import matplotlib.pyplot as plt

from stock_strategy_tester.data.data_loader import load_data, preprocess_data


def plot_moving_averages(data, short_window, long_window):
    """
    Plot the benchmark price and the moving averages on the same plot.

    :param data: DataFrame containing stock data with a 'Open' column.
    :param short_window: Lookback period for the short moving average.
    :param long_window: Lookback period for the long moving average.
    """
    if "Open" not in data.columns:
        raise ValueError("Input data must contain a 'Open' column.")

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

def plot_ptcv(data, short_window, long_window):
    if "Open" not in data.columns:
        raise ValueError("Input data must contain a 'Open' column.")

    # Calculate price and volume change percentage
    data["Close_pct"] = data["Close"].pct_change()
    data["Volume_pct"] = data["Volume"].pct_change()

    # Calculate PTCV
    data["ptcv"] = data["Close_pct"] * data["Volume_pct"]

    # Calculate short and long moving averages for PTCV
    data["SMA_Short"] = data["ptcv"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["ptcv"].rolling(window=long_window).mean()

    # Plot setup
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Close price on primary y-axis
    ax1.plot(data.index, data["Close"], label="Close Price", color="blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create secondary y-axis for PTCV moving averages
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["SMA_Short"], label=f"PTCV Short MA ({short_window} days)", color="red", linestyle="--")
    ax2.plot(data.index, data["SMA_Long"], label=f"PTCV Long MA ({long_window} days)", color="green", linestyle="--")
    ax2.set_ylabel("PTCV Moving Averages", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Combine legends
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Close Price and PTCV Moving Averages")
    plt.show()

# Example usage
if __name__ == "__main__":
    import yfinance as yf

    # Load sample data
    ticker = "JPM"
    # ticker = "AAPL"
    # ticker = "TSLA"
    start_date = "2017-01-01"
    end_date = "2024-01-01"
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Plot the benchmark price and moving averages
    plot_ptcv(data, short_window=5600, long_window=600)
    # plot_moving_averages(data, short_window=20, long_window=76)