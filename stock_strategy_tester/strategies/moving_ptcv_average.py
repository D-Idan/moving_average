import numpy as np
import pandas as pd


def ptcv_strategy(short_window=20, long_window=50, sides="long"):

    def strategy(data_s, fast_window=short_window, slow_window=long_window):
        """
        [Placeholder]
        """
        if "Close" not in data_s.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        # Calculate price and volume change percentage
        data_s["Close_pct"] = data_s["Close"].pct_change()
        data_s["Volume_pct"] = data_s["Volume"].pct_change()

        # Calculate ptcv
        data_s["ptcv"] = data_s["Close_pct"] * data_s["Volume_pct"]

        # Calculate short and long moving averages
        ptcv_Short = data_s["ptcv"].rolling(window=fast_window).mean()
        ptcv_Long = data_s["ptcv"].rolling(window=slow_window).mean()

        # Calculate the slope of the long moving average and volume
        ptcv_Long_Slope = ptcv_Long.diff()

        min_length = min(len(ptcv_Short), len(ptcv_Long), len(ptcv_Long_Slope))
        SMA_Short = ptcv_Short[-min_length:]
        SMA_Long = ptcv_Long[-min_length:]
        SMA_Long_Slope = ptcv_Long_Slope[-min_length:]

        # Determine crossover conditions
        long_condition = (SMA_Short > SMA_Long) # & (SMA_Long_Slope > 0)
        short_condition = (SMA_Short < SMA_Long) # & (SMA_Long_Slope < 0)

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Set the position based on the conditions
        position.loc[long_condition] = 1  # Long entry
        position.loc[short_condition] = -1  # Short entry

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

        # # Switch 1 to 0 and 0 to 1
        # long_signal = short_signal.copy()
        # # make short all 0
        # short_signal = short_signal * 0


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
