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
    confirm_days=2,
    long_diff=1,
    short_diff=1,
    reset_window=10,  # New parameter: Number of days to look back for price drops
    next_day_value=None,
):
    def strategy(data_s, alfa_short=alfa_short, alfa_long=alfa_long, short_window=short_window, long_window=long_window,
                 volume_power_short=volume_power_short, volume_power_long=volume_power_long, return_line=return_line, sides=sides,
                 next_day_execution=next_day_execution, confirm_days=confirm_days, long_diff=long_diff, short_diff=short_diff,
                 reset_window=reset_window, next_day_value=next_day_value

                 ):
        """
        Generate buy/sell signals with Exponential Moving VWAP (EMVWAP) crossovers and reset logic.
        """

        if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

        if next_day_value:
            # Add an additional row to avoid plotting the artificial line
            last_date = data_s.index[-1]
            next_date = last_date + pd.Timedelta(days=1)  # Increment the last date by one day
            additional_row = data_s.iloc[-1].copy()  # Duplicate the last row
            additional_row["Open"] = next_day_value  # Set the new index
            additional_row.name = next_date  # Set the new index
            data_s = pd.concat([data_s, pd.DataFrame([additional_row])])  # Append the new row


        # Helper: Calculate EM-VWAP
        def calculate_em_vwap(span, volume_power=100, day_price="Open", start_index=0):
            volume_power = volume_power / 100
            data_calc = data_s.iloc[start_index:]
            price_volume = data_calc[day_price] * (data_calc["Volume"] ** volume_power)
            ewma_price_volume = price_volume.ewm(span=span, adjust=False).mean()
            ewma_volume = data_calc["Volume"].ewm(span=span, adjust=False).mean()
            return ewma_price_volume / (ewma_volume ** volume_power)

        # Initialize EMVWAP values
        EMVWAP_Short = calculate_em_vwap(short_window, volume_power=volume_power_short, day_price="Open", start_index=0)
        EMVWAP_Long = calculate_em_vwap(long_window, volume_power=volume_power_long, day_price="Open", start_index=0)

        # Tracking minimum prices
        rolling_min_price = data_s["Open"].rolling(window=reset_window).min()

        # Identify rows where the reset condition occurs
        reset_condition = data_s["Open"] < rolling_min_price

        # Calculate EMVWAP values only at reset points
        reset_indices = np.where(reset_condition)[0]
        if len(reset_indices) > 0:
            # emvwap_short_reset = np.full_like(data_s["Close"], np.nan, dtype=float)
            emvwap_short_reset = EMVWAP_Short
            for idx in reset_indices:
                emvwap_short_reset[idx:] = calculate_em_vwap(
                    short_window,
                    volume_power=volume_power_short,
                    day_price="Open",
                    start_index=idx
                )
            EMVWAP_Short = pd.Series(emvwap_short_reset, index=data_s.index)
        else:
            EMVWAP_Short = calculate_em_vwap(
                short_window,
                volume_power=volume_power_short,
                day_price="Open",
                start_index=0
            )

        # Initialize position signals
        position = pd.Series(np.nan, index=data_s.index)

        long_line_condition = (EMVWAP_Long.diff(long_diff) > 0) & (data_s["Open"] > EMVWAP_Long)
        long_condition = (data_s["Open"] > EMVWAP_Short) & long_line_condition
        long_condition = long_condition & (EMVWAP_Short.diff() > 0.001)
        # short_condition = (data_s["Close"] < EMVWAP_Short) & (EMVWAP_Long.diff(long_diff) < 0)
        # long_condition = ((EMVWAP_Short.diff(long_diff) > 0) | (data_s["Close"] > EMVWAP_Short)) & (EMVWAP_Long.diff(long_diff) > 0)
        # short_condition = (EMVWAP_Short.diff(long_diff) < 0) & (EMVWAP_Long.diff(long_diff) < 0)
        # long_condition = (data_s["Close"] > EMVWAP_Short)
        # short_condition = (data_s["Close"] > EMVWAP_Short)
        # long_condition = (data_s["Close"] < EMVWAP_Short) & (EMVWAP_Long.diff(long_diff) > 0)
        # short_condition = (data_s["Close"] > EMVWAP_Short) & (EMVWAP_Long.diff(long_diff) < 0)
        short_condition = (EMVWAP_Long.diff(long_diff) < 0) & (data_s["Open"] < EMVWAP_Long)


        # Two-day confirmation for long_condition
        confirm_days = confirm_days
        confirmed_long_confirmation = long_condition.rolling(window=confirm_days).sum() == confirm_days
        confirmed_long_confirmation2 = ((data_s["Open"] < EMVWAP_Short) & long_line_condition).rolling(window=confirm_days).sum() == confirm_days
        confirmed_long_condition = (confirmed_long_confirmation | confirmed_long_confirmation2)
        # confirmed_long_condition = long_condition

        # Assign positions
        long_condition = confirmed_long_condition
        position[long_condition] = 1
        position[short_condition] = -1

        # Remove invalid sides
        if sides == "long":
            position[position < 0] = 0
        elif sides == "short":
            position[position > 0] = 0
        elif sides != "both":
            raise ValueError("Invalid value for 'sides'. Must be 'both', 'long', or 'short'.")

        # Forward-fill the positions to maintain them until the next change in the condition
        # position = position.ffill().fillna(0)
        position.loc[(long_condition == False) & (short_condition == False)] = 0  # No position if both are false
        position.loc[(long_condition == True) & (short_condition == True)] = 0  # No position if both are true
        # position.loc[(position == 1) & (data_s["Close"] < rolling_min_price)] = 0  # Reset long position if price drops

        # Return the positions as signals
        long_signal = (position == 1).astype(int)
        short_signal = (position == -1).astype(int)

        # Shift signals forward by one period for next-day execution
        if next_day_execution:
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