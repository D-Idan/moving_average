
def mean_reversion_strategy(window=20, num_std_dev=2, sides="both"):
    def strategy(data_s, rolling_window=window, std_dev=num_std_dev):
        """
        Generate buy/sell signals based on a mean reversion strategy using Bollinger Bands.

        :param data_s: DataFrame containing stock data with a 'Close' column.
        :param rolling_window: Lookback period for the rolling mean and standard deviation.
        :param std_dev: Number of standard deviations for the Bollinger Bands.
        :return: Series of signals: 1 for buy, -1 for sell, 0 for hold.
        """
        if "Close" not in data_s.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        # Calculate rolling mean and standard deviation
        rolling_mean = data_s["Close"].rolling(window=rolling_window).mean()
        rolling_std = data_s["Close"].rolling(window=rolling_window).std()

        # Calculate Bollinger Bands
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)

        # Generate buy signals when the price is below the lower band
        # Generate sell signals when the price is above the upper band
        data_s["long_signal"] = (data_s["Close"] < lower_band).astype(float)
        data_s["short_signal"] = (data_s["Close"] > upper_band).astype(float)

        # Adjust for the 'sides' parameter
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

        # Drop rows with NaN values resulting from the shift
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
    results = backtester.run(mean_reversion_strategy(window=20, num_std_dev=2))