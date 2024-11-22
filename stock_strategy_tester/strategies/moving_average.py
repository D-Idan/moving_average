import pandas as pd

def moving_average_strategy(short_window=20, long_window=50, sides="both"):

    def strategy(data_s, fast_window=short_window, slow_window=long_window):
        """
        Generate buy/sell signals based on a simple moving average (SMA) crossover strategy.

        :param data_s: DataFrame containing stock data with a 'Close' column.
        :param fast_window: Lookback period for the short moving average.
        :param slow_window: Lookback period for the long moving average.
        :return: Series of signals: 1 for buy, -1 for sell, 0 for hold.
        """
        if "Close" not in data_s.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        # Calculate short and long moving averages
        SMA_Short = data_s["Close"].rolling(window=fast_window).mean()
        SMA_Long = data_s["Close"].rolling(window=slow_window).mean()

        min_length = min(len(SMA_Short), len(SMA_Long))
        SMA_Short = SMA_Short[-min_length:]
        SMA_Long = SMA_Long[-min_length:]

        # Generate signals based on the moving average crossover
        if sides == "both":
            data_s["long_signal"] = (SMA_Short > SMA_Long).astype(float)
            data_s["short_signal"] = (SMA_Short < SMA_Long).astype(float)
        elif sides == "long":
            data_s["long_signal"] = (SMA_Short > SMA_Long).astype(float)
            data_s["short_signal"] = 0
        elif sides == "short":
            data_s["long_signal"] = 0
            data_s["short_signal"] = (SMA_Short < SMA_Long).astype(float)
        else:
            raise ValueError("Invalid value for 'sides'. Must be 'both', 'long', or 'short'.")


        # Fill NaN values with False and convert to integers
        data_s[["long_signal", "short_signal"]] = (
            data_s[["long_signal", "short_signal"]]
            .dropna()
            .astype(int)
        )

        # Debug prints
        # print(data_s["long_signal"].sum())
        # print(data_s["short_signal"].sum())
        # print((SMA_Short > SMA_Long).sum())
        # print((SMA_Short < SMA_Long).sum())

        return data_s["long_signal"], data_s["short_signal"]

    return strategy


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    from backtester.backtester import Backtester

    # Load sample data
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    data = yf.download(ticker, start=start_date, end=end_date)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the backtest
    results = backtester.run(moving_average_strategy(short_window=5, long_window=15))
    # Generate a report
    backtester.report()