import numpy as np
import pandas as pd


def moving_average_strategy(short_window=20, long_window=50, sides="both"):

    def strategy(data_s, fast_window=short_window, slow_window=long_window):
        """
        Generate buy/sell signals based on a simple moving average (SMA) crossover strategy,
        ensuring alignment with the slow moving average trend.

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

        # Calculate the slope of the long moving average and volume
        SMA_Long_Slope = SMA_Long.diff()
        SMA_Short_Slope = SMA_Short.diff()
        volume_slope = data_s["Volume"].rolling(window=5).mean().diff()

        min_length = min(len(SMA_Short), len(SMA_Long), len(SMA_Long_Slope), len(volume_slope))
        SMA_Short = SMA_Short[-min_length:]
        SMA_Long = SMA_Long[-min_length:]
        SMA_Long_Slope = SMA_Long_Slope[-min_length:]
        volume_slope = volume_slope[-min_length:]

        # Determine crossover conditions
        long_condition = (SMA_Short > SMA_Long) & (SMA_Long_Slope > 0)  # & (volume_slope > 0)
        short_condition = (SMA_Short < SMA_Long) & (SMA_Long_Slope < 0) # & (volume_slope < 0)
        volume_positive = volume_slope > 0

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Assign positions only at entry points use volume_positive
        long_entry = volume_positive & long_condition
        short_entry = volume_positive & short_condition
        position.loc[long_entry] = 1  # Long entry
        position.loc[short_entry] = -1  # Short entry

        # Forward-fill the positions to maintain them until the next change in the condition
        position = position.replace(0, np.nan)
        position.loc[(long_condition == False) & (short_condition == False)] = 0
        position = position.ffill().fillna(0)

        # Remove invalid sides
        if sides == "long":
            position[position < 0] = 0
        elif sides == "short":
            position[position > 0] = 0
        elif sides != "both":
            raise ValueError("Invalid value for 'sides'. Must be 'both', 'long', or 'short'.")

        # Return the positions as signals
        long_signal = (position == 1).astype(int)
        short_signal = (position == -1).astype(int)


        # # # Shift signals forward by one period for next-day execution
        # # 15 / 5 is the best if not using the shift
        # # Note: It makes much more sense to shift the signals after the side is chosen
        # #       But it makes the results much worse
        # #       So I will shift the signals before the side is chosen
        # #       Means that I need to calculate the enter and exit signals in the same day
        # # data_s["long_signal"] = long_signal.shift(1)
        # # data_s["short_signal"] = short_signal.shift(1)

        long_signal = long_signal.shift(1)
        short_signal = short_signal.shift(1)

        # Drop any rows where signals are NaN due to the shift
        data_s["long_signal"] = long_signal
        data_s["short_signal"] = short_signal
        data_s.dropna(inplace=True)

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
    results = backtester.run(moving_average_strategy(short_window=5, long_window=15))
