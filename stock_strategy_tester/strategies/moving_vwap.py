import numpy as np
import pandas as pd


def moving_vwap_strategy(short_window=20, long_window=50, sides="both"):

    def strategy(data_s, fast_window=short_window, slow_window=long_window):
        """
        Generate buy/sell signals based on a moving VWAP crossover strategy,
        ensuring alignment with the long VWAP trend.

        :param data_s: DataFrame containing stock data with 'Close', 'Volume', and 'High'/'Low' columns.
        :param fast_window: Lookback period for the short moving VWAP.
        :param slow_window: Lookback period for the long moving VWAP.
        :return: Series of signals: 1 for buy, -1 for sell, 0 for hold.
        """
        if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

        # Calculate VWAP for each period
        def calculate_vwap(window):
            price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
            cum_price_volume = price_volume.rolling(window=window).sum()
            cum_volume = data_s["Volume"].rolling(window=window).sum()
            return cum_price_volume / cum_volume

        VWAP_Short = calculate_vwap(fast_window)
        VWAP_Long = calculate_vwap(slow_window)

        # Calculate the slope of the long VWAP
        VWAP_Long_Slope = VWAP_Long.diff()

        min_length = min(len(VWAP_Short), len(VWAP_Long), len(VWAP_Long_Slope))
        VWAP_Short = VWAP_Short[-min_length:]
        VWAP_Long = VWAP_Long[-min_length:]
        VWAP_Long_Slope = VWAP_Long_Slope[-min_length:]

        # Determine crossover conditions
        long_condition = (VWAP_Short > VWAP_Long) & (VWAP_Long_Slope > 0)
        short_condition = (VWAP_Short < VWAP_Long) & (VWAP_Long_Slope < 0)

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Assign positions only at entry points
        long_entry = long_condition
        short_entry = short_condition
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

        # Shift signals forward by one period for next-day execution
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
    results = backtester.run(moving_vwap_strategy(short_window=5, long_window=15))