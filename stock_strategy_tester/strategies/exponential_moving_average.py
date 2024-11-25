
def exponential_moving_average_strategy(short_window=20, long_window=50, sides="both"):
    """
    Generate buy/sell signals based on an Exponential Moving Average (EMA) crossover strategy,
    ensuring alignment with the slow EMA trend.

    :param short_window: Lookback period for the short EMA.
    :param long_window: Lookback period for the long EMA.
    :param sides: Type of trades: 'both', 'long', or 'short'.
    :return: Strategy function.
    """
    def strategy(data_s, fast_window=short_window, slow_window=long_window):
        """
        Generate buy/sell signals based on EMA crossover strategy.

        :param data_s: DataFrame containing stock data with a 'Close' column.
        :param fast_window: Lookback period for the short EMA.
        :param slow_window: Lookback period for the long EMA.
        :return: Series of signals: 1 for buy, -1 for sell, 0 for hold.
        """
        if "Close" not in data_s.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        # Calculate short and long exponential moving averages
        EMA_Short = data_s["Close"].ewm(span=fast_window, adjust=False).mean()
        EMA_Long = data_s["Close"].ewm(span=slow_window, adjust=False).mean()

        # Calculate the slope of the long EMA
        EMA_Long_Slope = EMA_Long.diff()

        # Truncate all series to the minimum length
        min_length = min(len(EMA_Short), len(EMA_Long), len(EMA_Long_Slope))
        EMA_Short = EMA_Short[-min_length:]
        EMA_Long = EMA_Long[-min_length:]
        EMA_Long_Slope = EMA_Long_Slope[-min_length:]

        # Generate initial signals based on the EMA crossover
        data_s["long_signal"] = (EMA_Short > EMA_Long).astype(float)
        data_s["short_signal"] = (EMA_Short < EMA_Long).astype(float)

        # Adjust signals based on the slow EMA trend
        data_s["long_signal"] = data_s["long_signal"] * (EMA_Long_Slope > 0).astype(float)
        data_s["short_signal"] = data_s["short_signal"] * (EMA_Long_Slope < 0).astype(float)

        # Shift signals forward by one period for next-day execution
        data_s["long_signal"] = data_s["long_signal"].shift(1)
        data_s["short_signal"] = data_s["short_signal"].shift(1)

        # Drop any rows where signals are NaN due to the shift
        data_s.dropna(inplace=True)

        # Handle different 'sides' options
        if sides == "both":
            pass
        elif sides == "long":
            data_s["short_signal"] = 0
        elif sides == "short":
            data_s["long_signal"] = 0
        else:
            raise ValueError("Invalid value for 'sides'. Must be 'both', 'long', or 'short'.")

        # Fill NaN values with False and convert to integers
        data_s[["long_signal", "short_signal"]] = (
            data_s[["long_signal", "short_signal"]]
            .dropna()
            .astype(int)
        )

        return data_s["long_signal"], data_s["short_signal"]

    return strategy


if __name__ == "__main__":
    # Example usage
    from backtester.backtester import Backtester
    from data.data_loader import load_data, preprocess_data

    # Load sample data
    ticker = "AAPL"
    start_date = "2017-01-01"
    end_date = "2021-01-01"
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the backtest
    results = backtester.run(exponential_moving_average_strategy(short_window=5, long_window=15))