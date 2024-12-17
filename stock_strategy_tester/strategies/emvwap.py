import numpy as np
import pandas as pd

from backtester.performance import generate_report_backtest


def emvwap_strategy(short_window=63, long_window=63*4, alfa_short=50, alfa_long=50, sides="long"):

    def strategy(data_s, alfa_short=alfa_short, alfa_long=alfa_long, short_window=short_window, long_window=long_window, return_line=False):
        """
        Generate buy/sell signals based on Exponential Moving VWAP (EMVWAP) crossovers.
        """
        if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

        # Helper: Calculate VWAP and Moving VWAP (MVWAP)
        def calculate_mvwav(window):
            # price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
            price_volume = data_s["Open"] * data_s["Volume"]
            rolling_price_volume = price_volume.rolling(window=window).sum()
            rolling_volume = data_s["Volume"].rolling(window=window).sum()
            return rolling_price_volume / rolling_volume

        # Helper: Calculate EM-VWAP
        def calculate_em_vwap(span):
            # price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
            price_volume = data_s["Open"] * data_s["Volume"]
            ewma_price_volume = price_volume.ewm(span=span, adjust=False).mean()
            ewma_volume = data_s["Volume"].ewm(span=span, adjust=False).mean()
            return ewma_price_volume / ewma_volume

        # Calculate indicators
        EMVWAP_Short = calculate_em_vwap(short_window)
        EMVWAP_Long = calculate_em_vwap(long_window)

        # Detect where the slope of EMVWAP_Long changes from positive to negative
        slope_long_emvwap = EMVWAP_Long.diff()
        # signal_slope_change = (slope_long_emvwap < 0) & (slope_long_emvwap.shift(1) >= 0)


        min_length = min(len(EMVWAP_Long), len(EMVWAP_Short), len(slope_long_emvwap))
        EMVWAP_Long = EMVWAP_Long[-min_length:]
        EMVWAP_Short = EMVWAP_Short[-min_length:]
        slope_long_emvwap = slope_long_emvwap[-min_length:]
        price = data_s["Open"].copy()[-min_length:]

        # Determine conditions
        # Note:
        # Slope is not useful

        # Long condition
        alfa_long = alfa_long / 100       # LONG 0.65
        EMVWAP_long_calc = alfa_long * EMVWAP_Short + (1 - alfa_long) * EMVWAP_Long
        long_condition = price > EMVWAP_long_calc

        # Short condition
        alfa_short = alfa_short / 100       # SHORT 0.65
        EMVWAP_short_calc = alfa_short * EMVWAP_Short + (1 - alfa_short) * EMVWAP_Long
        short_condition = price < EMVWAP_short_calc

        # Ensure conditions last at least # days
        days = 2
        long_condition = long_condition.rolling(window=days).sum() == days
        short_condition = short_condition.rolling(window=days).sum() == days

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Assign positions only at entry points
        long_entry = long_condition
        short_entry = short_condition
        position.loc[long_entry] = 1  # Long entry
        position.loc[short_entry] = -1  # Short entry
        position.loc[(long_entry == True) & (short_entry == True)] = 0  # No position if both are true

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

        if return_line:
            lines = {'long': EMVWAP_long_calc, 'short': EMVWAP_short_calc}
            return data_s["long_signal"], data_s["short_signal"], lines

        return data_s["long_signal"], data_s["short_signal"]

    return strategy


if __name__ == "__main__":
    # Example usage
    from backtester.backtester import Backtester
    from data.data_loader import load_data, preprocess_data

    # Load sample data
    ticker = "JPM"
    # ticker = "AAPL"
    start_date = "2017-01-01"
    end_date = "2021-01-01"
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the backtest
    results = backtester.run(emvwap_strategy(short_window=5, long_window=15))

    generate_report_backtest(results['data'])