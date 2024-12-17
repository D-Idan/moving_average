import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_emvwap_mvwav_ema(data_s, short_window, long_window):
    """
    Plot EM-VWAP, MVWAP, and EMA on the same chart, with crossover points.

    :param data_s: DataFrame containing the stock data.
    :param short_window: Lookback period for the short-term moving averages.
    :param long_window: Lookback period for the long-term moving averages.
    """

    if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
        raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

    # Helper: Calculate VWAP and Moving VWAP (MVWAP)
    def calculate_mvwav(window):
        price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
        price_volume = data_s["Open"] * data_s["Volume"]
        rolling_price_volume = price_volume.rolling(window=window).sum()
        rolling_volume = data_s["Volume"].rolling(window=window).sum()
        return rolling_price_volume / rolling_volume

    # Helper: Calculate EM-VWAP
    def calculate_em_vwap(span):
        price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
        price_volume = data_s["Open"]  * data_s["Volume"]
        ewma_price_volume = price_volume.ewm(span=span, adjust=False).mean()
        ewma_volume = data_s["Volume"].ewm(span=span, adjust=False).mean()
        return ewma_price_volume / ewma_volume

    # Calculate indicators
    EMVWAP_Short = calculate_em_vwap(short_window)
    EMVWAP_Long = calculate_em_vwap(long_window)
    MVWAP_Short = calculate_mvwav(short_window)
    MVWAP_Long = calculate_mvwav(long_window)
    EMA_Short = data_s["Close"].ewm(span=short_window, adjust=False).mean()
    EMA_Long = data_s["Close"].ewm(span=long_window, adjust=False).mean()

    # Detect where the slope of EMVWAP_Long changes from positive to negative
    slope_long_emvwap = EMVWAP_Long.diff()
    signal_slope_change = (slope_long_emvwap < 0) & (slope_long_emvwap.shift(1) >= 0)

    # Detect crossover points for all indicators
    crossover_emvwap = (EMVWAP_Short > EMVWAP_Long) & (EMVWAP_Short.shift(1) <= EMVWAP_Long.shift(1)) | \
                       (EMVWAP_Short < EMVWAP_Long) & (EMVWAP_Short.shift(1) >= EMVWAP_Long.shift(1))

    crossover_mvwav = (MVWAP_Short > MVWAP_Long) & (MVWAP_Short.shift(1) <= MVWAP_Long.shift(1)) | \
                      (MVWAP_Short < MVWAP_Long) & (MVWAP_Short.shift(1) >= MVWAP_Long.shift(1))

    crossover_ema = (EMA_Short > EMA_Long) & (EMA_Short.shift(1) <= EMA_Long.shift(1)) | \
                    (EMA_Short < EMA_Long) & (EMA_Short.shift(1) >= EMA_Long.shift(1))

    # Extract crossover dates and prices
    crossover_dates_emvwap = data_s.index[crossover_emvwap]
    crossover_prices_emvwap = data_s["Close"][crossover_emvwap]

    crossover_dates_mvwav = data_s.index[crossover_mvwav]
    crossover_prices_mvwav = data_s["Close"][crossover_mvwav]

    crossover_dates_ema = data_s.index[crossover_ema]
    crossover_prices_ema = data_s["Close"][crossover_ema]

    # Plot setup
    plt.figure(figsize=(16, 9))
    plt.plot(data_s.index, data_s["Close"], label="Close Price", color="blue", alpha=0.5)

    # Format the x-axis to show dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show major ticks every 3 months
    plt.gcf().autofmt_xdate()  # Automatically format the date labels to prevent overlap

    # Plot EM-VWAP
    plt.plot(data_s.index, EMVWAP_Short, label=f"EMVWAP Short ({short_window})", color="red", linestyle="--")
    plt.plot(data_s.index, EMVWAP_Long, label=f"EMVWAP Long ({long_window})", color="darkred", linestyle="--")

    # # Plot MVWAP
    plt.plot(data_s.index, MVWAP_Short, label=f"MVWAP Short ({short_window})", color="orange", linestyle="-")
    # plt.plot(data_s.index, MVWAP_Long, label=f"MVWAP Long ({long_window})", color="darkorange", linestyle="-")

    # # Plot EMA
    # plt.plot(data_s.index, EMA_Short, label=f"EMA Short ({short_window})", color="green", linestyle=":")
    plt.plot(data_s.index, EMA_Long, label=f"EMA Long ({long_window})", color="darkgreen", linestyle=":")

    ### Add crossover points
    # plt.scatter(crossover_dates_emvwap, crossover_prices_emvwap, color="red", label="EMVWAP Crossovers", alpha=0.8,
    #             marker="x")
    # plt.scatter(crossover_dates_mvwav, crossover_prices_mvwav, color="orange", label="MVWAP Crossovers", alpha=0.8,
    #             marker="o")
    # plt.scatter(crossover_dates_ema, crossover_prices_ema, color="green", label="EMA Crossovers", alpha=0.8, marker="^")

    # Plot slope change signal
    crossover_dates_slope = data_s.index[signal_slope_change]
    crossover_prices_slope = data_s["Close"][signal_slope_change]
    plt.scatter(crossover_dates_slope, crossover_prices_slope, color="purple", label="Slope Change Signal", alpha=0.8,
                marker="s")

    # Plot annotations
    plt.title("EM-VWAP, MVWAP, and EMA Indicators with Crossovers")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    from data.data_loader import load_data, preprocess_data
    import yfinance as yf
    # Date
    from datetime import datetime

    # Load sample data
    # ticker = "VYM"
    ticker = "JPM"
    # ticker = "AAPL"
    # ticker = "TSLA"
    # ticker = "SPY"
    # ticker = "SQ"
    # ticker = "SPMO"
    start_date = "2017-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Plot the EM-VWAP, MVWAP, and EMA indicators
    data.index = pd.to_datetime(data["Date"])
    plot_emvwap_mvwav_ema(data, short_window=63, long_window=63*4)