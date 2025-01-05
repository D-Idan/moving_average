import numpy as np
import pandas as pd

from backtester.performance import generate_report_backtest


def emvwap_strategy_with_reset(
    short_window=63,
    long_window=63*4,
    alfa_short=50,
    alfa_long=50,
    volume_power_short=100,
    volume_power_long=100,
    sides="long",
    next_day_execution=True,
    return_line=False,
    stop_loss_days=1,
    long_diff=1,
    short_diff=1,
    reset_window=10  # New parameter: Number of days to look back for price drops
):
    def strategy(data_s, **kwargs):
        """
        Generate buy/sell signals with Exponential Moving VWAP (EMVWAP) crossovers and reset logic.
        """
        # Get parameters
        short_window = kwargs.get("short_window", 63)
        long_window = kwargs.get("long_window", 63 * 4)
        volume_power_short = kwargs.get("volume_power_short", 100)
        volume_power_long = kwargs.get("volume_power_long", 100)
        sides = kwargs.get("sides", "long")
        next_day_execution = kwargs.get("next_day_execution", True)
        return_line = kwargs.get("return_line", False)
        stop_loss_days = kwargs.get("stop_loss_days", 1)
        long_diff = kwargs.get("long_diff", 1)
        short_diff = kwargs.get("short_diff", 1)
        reset_window = kwargs.get("reset_window", 10)

        if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

        # Helper: Calculate EM-VWAP
        def calculate_em_vwap(span, volume_power=100, day_price="Close", start_index=0):
            volume_power = volume_power / 100
            data_calc = data_s.iloc[start_index:]
            price_volume = data_calc[day_price] * (data_calc["Volume"] ** volume_power)
            ewma_price_volume = price_volume.ewm(span=span, adjust=False).mean()
            ewma_volume = data_calc["Volume"].ewm(span=span, adjust=False).mean()
            return ewma_price_volume / (ewma_volume ** volume_power)

        # Initialize EMVWAP values
        EMVWAP_Short = calculate_em_vwap(short_window, volume_power=volume_power_short, day_price="High", start_index=0)
        EMVWAP_Long = calculate_em_vwap(long_window, volume_power=volume_power_long, day_price="Low", start_index=0)

        # Initialize variables for reset logic
        last_reset_index = None

        # Tracking minimum prices
        rolling_min_price = data_s["Open"].rolling(window=reset_window).min()

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        for idx in range(len(data_s)):
            if last_reset_index is None or idx >= last_reset_index:
                # Get current data slice
                current_price = data_s["Close"].iloc[idx]

                # Check for reset condition
                if current_price < rolling_min_price.iloc[idx]:
                    last_reset_index = idx
                    EMVWAP_Short[idx:] = calculate_em_vwap(short_window, volume_power=volume_power_short, day_price="High", start_index=idx)
                    continue

                long_condition = (current_price > EMVWAP_Short.iloc[idx]) & (EMVWAP_Long.diff(long_diff).iloc[idx] > 0)
                short_condition = (current_price < EMVWAP_Short.iloc[idx]) & (EMVWAP_Long.diff(long_diff).iloc[idx] < 0)

                # Assign positions
                if long_condition:
                    position.iloc[idx] = 1
                elif short_condition:
                    position.iloc[idx] = -1

        # Remove invalid sides
        if sides == "long":
            position[position < 0] = 0
        elif sides == "short":
            position[position > 0] = 0
        elif sides != "both":
            raise ValueError("Invalid value for 'sides'. Must be 'both', 'long', or 'short'.")

        # Forward-fill the positions to maintain them until the next change in the condition
        position = position.replace(0, np.nan)
        position = position.ffill().fillna(0)
        position.loc[(position.shift(1) == 1) & (data_s["Close"] < rolling_min_price)] = 0  # Reset long position if price drops

        # Return the positions as signals
        long_signal = (position == 1).astype(int)
        short_signal = (position == -1).astype(int)

        # Shift signals forward by one period for next-day execution
        if next_day_execution:
            long_signal = long_signal.shift(stop_loss_days)
            short_signal = short_signal.shift(stop_loss_days)

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
            lines = {'long': EMVWAP_Long, 'short': EMVWAP_Short}
            return data_s["long_signal"], data_s["short_signal"], lines

        return data_s["long_signal"], data_s["short_signal"]

    return strategy


if __name__ == "__main__":
    # Example usage
    from backtester.backtester import Backtester
    from data.data_loader import load_data, preprocess_data

    # Load sample data
    ticker = "JPM"
    start_date = "2017-01-01"
    end_date = "2021-01-01"
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the backtest
    results = backtester.run(emvwap_strategy_with_reset(short_window=5, long_window=15, reset_window=10))

    generate_report_backtest(results['data'])