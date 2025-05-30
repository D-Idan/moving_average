import numpy as np
import pandas as pd

from backtester.performance import generate_report_backtest


def emvwap_strategy(short_window=63, long_window=63*4, alfa_short=50, alfa_long=50, volume_power_short=100,
                    volume_power_long=100, sides="long", next_day_execution=True, return_line=False, stop_loss_days=300,
                    long_diff=1, short_diff=1

                    ):

    def strategy(data_s, alfa_short=alfa_short, alfa_long=alfa_long, short_window=short_window, long_window=long_window,
                 volume_power_short=volume_power_short, volume_power_long=volume_power_long, return_line=return_line, sides=sides,
                 next_day_execution=next_day_execution, stop_loss_days=stop_loss_days, long_diff=long_diff, short_diff=short_diff

                 ):
        """
        Generate buy/sell signals based on Exponential Moving VWAP (EMVWAP) crossovers.
        """
        if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

        # Helper: Calculate VWAP and Moving VWAP (MVWAP)
        def calculate_mvwav(window, volume_power=100):
            volume_power = volume_power / 100
            # price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
            price_volume = data_s["Open"] * data_s["Volume"] ** volume_power
            rolling_price_volume = price_volume.rolling(window=window).sum()
            rolling_volume = data_s["Volume"].rolling(window=window).sum()
            return rolling_price_volume / rolling_volume ** volume_power

        # Helper: Calculate EM-VWAP
        def calculate_em_vwap(span, volume_power=100, day_price="Close"):
            volume_power = volume_power / 100
            # price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
            price_volume = data_s[day_price] * (data_s["Volume"] ** volume_power)
            ewma_price_volume = price_volume.ewm(span=span, adjust=False).mean()
            ewma_volume = data_s["Volume"].ewm(span=span, adjust=False).mean()
            return ewma_price_volume / (ewma_volume ** volume_power)

        # Calculate indicators
        EMVWAP_Short = calculate_em_vwap(short_window, volume_power=volume_power_short, day_price="High")
        EMVWAP_Long = calculate_em_vwap(long_window, volume_power=volume_power_long, day_price="Low")


        min_length = min(len(EMVWAP_Long), len(EMVWAP_Short))
        EMVWAP_Long = EMVWAP_Long[-min_length:]
        EMVWAP_Short = EMVWAP_Short[-min_length:]
        slope_long_emvwap = EMVWAP_Long.diff(long_diff)[-min_length:] # 64 is the window TODO: Change to 1
        slope_short_emvwap = EMVWAP_Short.diff(short_diff)[-min_length:] # 20 is the window TODO: Change to 1
        price = data_s["High"].copy()[-min_length:]

        # Determine conditions
        # Note:
        # Slope is not useful

        # Long condition
        alfa_long = alfa_long / 100       # LONG 0.65
        EMVWAP_long_calc = alfa_long * price + (1 - alfa_long) * EMVWAP_Long
        long_condition = (price * 0.985 > EMVWAP_long_calc) & (EMVWAP_long_calc.diff(long_diff) > 0)
        long_condition = (EMVWAP_long_calc.diff(long_diff) >= 0) & (price > EMVWAP_Short) # TODO: Change <to 0.985

        # Short condition
        alfa_short = alfa_short / 100       # SHORT 0.65
        EMVWAP_short_calc = alfa_short * price + (1 - alfa_short) * EMVWAP_Short
        short_condition = (price * 1.015 < EMVWAP_short_calc) #& (EMVWAP_short_calc.diff(short_diff) < 0) # TODO: Change <to 1.015
        short_condition = (EMVWAP_long_calc.diff(long_diff) < 0) & (price < EMVWAP_Short) # TODO: Change <to 0.985


        # Ensure conditions last at least # days
        days = 2
        long_condition = long_condition.rolling(window=days).sum() == days
        short_condition = (short_condition.rolling(window=days).sum() == days)


        # # # STOP-LOSS (If the price in the current trade gets lower in 2% from the maximum price, close the trade)
        # stop_loss_flag = ~(price < price.rolling(window=stop_loss_days).max() * 0.98)
        # sl_2_condition = (price > price.rolling(window=5).mean()) & stop_loss_flag
        # long_condition = long_condition & sl_2_condition
        # # short_condition = short_condition & (price < price.rolling(window=stop_loss_days).min() * 1.02)

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Assign positions only at entry points
        long_entry = long_condition
        short_entry = short_condition
        position.loc[long_entry] = 1  # Long entry
        position.loc[short_entry] = -1  # Short entry

        # Forward-fill the positions to maintain them until the next change in the condition
        position = position.replace(0, np.nan)
        position = position.ffill().fillna(0)
        position.loc[(long_condition == False) & (short_condition == False)] = 0 # No position if both are false
        position.loc[(long_entry == True) & (short_entry == True)] = 0  # No position if both are true

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