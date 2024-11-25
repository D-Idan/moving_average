
def ichimoku_cloud_strategy(short_window=20, long_window=50, sides="both"):
    """
    Generate buy/sell signals based on the Ichimoku Cloud strategy.

    :param sides: Type of trades: 'both', 'long', or 'short'.
    :return: Strategy function.
    """
    def strategy(data_s):
        """
        Generate buy/sell signals based on the Ichimoku Cloud strategy.

        :param data_s: DataFrame containing stock data with 'High', 'Low', and 'Close' columns.
        :return: Series of signals: 1 for buy, -1 for sell, 0 for hold.
        """
        if not {"High", "Low", "Close"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'High', 'Low', and 'Close' columns.")

        # Calculate Ichimoku Cloud components
        high_9 = data_s["High"].rolling(window=short_window).max()
        low_9 = data_s["Low"].rolling(window=short_window).min()
        data_s["tenkan_sen"] = (high_9 + low_9) / 2  # Conversion Line (9-period average)

        high_26 = data_s["High"].rolling(window=long_window).max()
        low_26 = data_s["Low"].rolling(window=long_window).min()
        data_s["kijun_sen"] = (high_26 + low_26) / 2  # Base Line (26-period average)

        data_s["senkou_span_a"] = ((data_s["tenkan_sen"] + data_s["kijun_sen"]) / 2).shift(long_window)  # Leading Span A

        high_52 = data_s["High"].rolling(window=(long_window*2)).max()
        low_52 = data_s["Low"].rolling(window=(long_window*2)).min()
        data_s["senkou_span_b"] = ((high_52 + low_52) / 2).shift(long_window)  # Leading Span B

        data_s["chikou_span"] = data_s["Close"].shift(-long_window)  # Lagging Span

        # Generate buy/sell signals
        data_s["long_signal"] = (
            (data_s["Close"] > data_s["senkou_span_a"]) &
            (data_s["Close"] > data_s["senkou_span_b"]) &
            (data_s["tenkan_sen"] > data_s["kijun_sen"])
        ).astype(float)

        data_s["short_signal"] = (
            (data_s["Close"] < data_s["senkou_span_a"]) &
            (data_s["Close"] < data_s["senkou_span_b"]) &
            (data_s["tenkan_sen"] < data_s["kijun_sen"])
        ).astype(float)

        # Handle different 'sides' options
        if sides == "both":
            pass
        elif sides == "long":
            data_s["short_signal"] = 0
        elif sides == "short":
            data_s["long_signal"] = 0
        else:
            raise ValueError("Invalid value for 'sides'. Must be 'both', 'long', or 'short'.")

        # Shift signals forward by one period for next-day execution
        data_s["long_signal"] = data_s["long_signal"].shift(1)
        data_s["short_signal"] = data_s["short_signal"].shift(1)

        # Drop NaN values due to rolling calculations and shifts
        data_s.dropna(inplace=True)

        # Convert signals to integers
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
    results = backtester.run(ichimoku_cloud_strategy())