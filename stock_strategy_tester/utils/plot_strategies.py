import matplotlib.pyplot as plt

from data.data_loader import load_data, preprocess_data


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


import matplotlib.pyplot as plt


def plot_ptcv(data_s, short_window, long_window):
    if "Close" not in data_s.columns:
        raise ValueError("Input data must contain a 'Close' column.")
    if "Volume" not in data_s.columns:
        raise ValueError("Input data must contain a 'Volume' column.")

    # Calculate price and volume change percentage
    data_s["Close_pct"] = data_s["Close"].pct_change().rolling(window=short_window).mean()
    data_s["Volume_pct"] = data_s["Volume"].pct_change().rolling(window=long_window).mean()

    # Calculate ptcv
    data_s["ptcv"] = data_s["Close_pct"] * data_s["Volume_pct"]

    # Calculate short and long moving averages
    ptcv_Short = data_s["ptcv"].rolling(window=short_window).mean()
    ptcv_Long = data_s["ptcv"].rolling(window=long_window).mean()

    # Detect crossovers
    crossover_points = (ptcv_Short > ptcv_Long) & (ptcv_Short.shift(1) <= ptcv_Long.shift(1)) | \
                       (ptcv_Short < ptcv_Long) & (ptcv_Short.shift(1) >= ptcv_Long.shift(1))

    crossover_dates = data_s.index[crossover_points]
    crossover_prices = data_s["Close"][crossover_points]

    # Plot setup
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Close price on primary y-axis
    ax1.plot(data_s.index, data_s["Close"], label="Close Price", color="blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create secondary y-axis for PTCV moving averages
    ax2 = ax1.twinx()
    ax2.plot(data_s.index, ptcv_Short, label=f"PTCV Short MA ({short_window} days)", color="red", linestyle="--")
    ax2.plot(data_s.index, ptcv_Long, label=f"PTCV Long MA ({long_window} days)", color="green", linestyle="--")
    ax2.set_ylabel("PTCV Moving Averages", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Add vertical lines and annotate crossover points
    for date, price in zip(crossover_dates, crossover_prices):
        ax1.axvline(x=date, color='purple', linestyle='--', linewidth=0.8)
        ax1.annotate(f'{price:.2f}', xy=(date, price), xytext=(date, price + price * 0.02),
                     arrowprops=dict(arrowstyle='->', color='purple'), fontsize=9, color='purple')

    # Combine legends
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Close Price and PTCV Moving Averages with Crossovers")
    plt.show()

# Example usage
if __name__ == "__main__":
    import yfinance as yf

    # Load sample data
    ticker = "VYM"
    # ticker = "JPM"
    # ticker = "AAPL"
    # ticker = "TSLA"
    start_date = "2017-01-01"
    end_date = "2024-01-01"
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Plot the benchmark price and moving averages
    plot_ptcv(data, short_window=70, long_window=80)
    # plot_moving_averages(data, short_window=20, long_window=76)