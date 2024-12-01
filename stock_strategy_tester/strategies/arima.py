import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arma_strategy(low=4, high=4, threshold=0.01, sides="both"):
# def arma_strategy(order=(4, 0, 4), threshold=0.01, sides="both"):
    """
    Generate buy/sell signals based on an ARMA model.

    :param order: ARMA model order (p, d, q).
    :param threshold: Minimum predicted return for generating signals.
    :param sides: 'both', 'long', or 'short' indicating which signals to consider.
    :return: Function that returns buy/sell signals.
    """
    order = (low, 0, high)
    def strategy(data_s):
        if "Close" not in data_s.columns:
            raise ValueError("Input data must contain a 'Close' column.")

        # Calculate returns
        data_s["Return"] = data_s["Close"].pct_change().fillna(0)

        # Fit ARMA model
        model = ARIMA(data_s["Return"], order=order).fit()

        # Predict returns
        data_s["Predicted_Return"] = model.predict(start=order[0], end=len(data_s) - 1)

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Generate signals based on predicted returns and threshold
        long_entry = data_s["Predicted_Return"] > threshold
        short_entry = data_s["Predicted_Return"] < -threshold
        position.loc[long_entry] = 1  # Long entry
        position.loc[short_entry] = -1  # Short entry

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
    from backtester.backtester import Backtester
    from data.data_loader import load_data, preprocess_data

    # Load sample data
    ticker = "AAPL"
    start_date = "2017-01-01"
    end_date = "2021-01-01"
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the backtest
    results = backtester.run(arma_strategy(order=(4, 0, 4), threshold=0.01))