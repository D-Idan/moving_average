import numpy as np
import pandas as pd

from backtester.performance import generate_report_backtest


def donchian_avarage_strategy(short_window=63, long_window=63*4, alfa_short=50, alfa_long=50, volume_power_short=100,
                    volume_power_long=100, sides="long", next_day_execution=True, return_line=False, stop_loss_days=3,
                    long_diff=1, short_diff=1

                    ):

    def strategy(data_s, alfa_short=alfa_short, alfa_long=alfa_long, short_window=short_window, long_window=long_window,
                 volume_power_short=volume_power_short, volume_power_long=volume_power_long, return_line=return_line, sides=sides,
                 next_day_execution=next_day_execution, stop_loss_days=stop_loss_days, long_diff=long_diff, short_diff=short_diff

                 ):
        """
        Generate buy/sell signals based on Exponential Moving VWAP (EMVWAP) and Donchian strategy
        """
        if not {"Close", "Volume", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Volume', 'High', and 'Low' columns.")

        # Helper: Calculate Donchian Channel
        def calculate_donchian(window):
            high = data_s["High"].rolling(window=window).max()
            low = data_s["Low"].rolling(window=window//2).min()
            results = {"high": high, "low": low, "mid": (high + low) / 2}
            return pd.DataFrame(results)

        # Helper: Calculate EM-VWAP
        def calculate_em_vwap(span, volume_power=100, day_price="Close"):
            volume_power = volume_power / 100
            price_volume = (data_s["High"] + data_s["Low"] + data_s["Close"]) / 3 * data_s["Volume"]
            price_volume = data_s[day_price] * (data_s["Volume"] ** volume_power)
            ewma_price_volume = price_volume.ewm(span=span, adjust=False).mean()
            ewma_volume = data_s["Volume"].ewm(span=span, adjust=False).mean()
            return ewma_price_volume / (ewma_volume ** volume_power)
            # return data_s[day_price].ewm(span=span, adjust=False).mean()

        # Calculate indicators
        donchian = calculate_donchian(short_window)
        EMVWAP_Long = calculate_em_vwap(long_window, volume_power=volume_power_long, day_price="Low")
        EMVWAP_Short = calculate_em_vwap(short_window, volume_power=volume_power_short, day_price="High")
        EMVWAP_Long_diff = EMVWAP_Long.diff(long_diff)


        min_length = min(len(EMVWAP_Long), len(EMVWAP_Long_diff), len(donchian))
        EMVWAP_Long = EMVWAP_Long[-min_length:]
        EMVWAP_Short = EMVWAP_Short[-min_length:]
        price = data_s["Open"].copy()[-min_length:]
        donchian = donchian[-min_length:]

        # Determine conditions
        # Note:

        # EMVWAP long condition
        alfa_long = alfa_long / 100
        EMVWAP_long_calc = (1 - alfa_long) * EMVWAP_Long + (alfa_long) * EMVWAP_Short
        long_condition1 = (EMVWAP_long_calc.diff(long_diff) > 0.0)
        long_condition1 = long_condition1 & (price > EMVWAP_Short)

        # donchian Short condition
        donchian["Signal_long"] = float('nan')
        donchian["Signal_short"] = float('nan')

        alfa_short = (alfa_short / 1000) + 1

        # Exit signal: Sell when Close crosses below the Lower band
        lower_cond_l = (price > alfa_short * donchian['low'].shift(1)) #| (price > EMVWAP_long_calc)
        lower_cond_s = alfa_short * price > donchian['high'].shift(1)
        donchian.loc[lower_cond_l, 'Signal_long'] = 1
        donchian.loc[lower_cond_s, 'Signal_short'] = 1

        # # Entry signal: Buy when Close crosses above the Upper band
        # upper_cond = price > (2-alfa_short) * donchian['mid'].shift(1)
        # donchian.loc[upper_cond, 'Signal_long'] = 1
        # donchian.loc[~upper_cond, 'Signal_short'] = 1


        donchian["Signal_long"] = donchian["Signal_long"].fillna(0)
        donchian["Signal_short"] = donchian["Signal_short"].fillna(0)


        # ONLY LONG
        long_condition = long_condition1 & donchian["Signal_long"]
        short_condition = ~long_condition1 & donchian["Signal_short"]

        # Ensure conditions last at least # days
        days = stop_loss_days
        long_condition = long_condition.rolling(window=days).sum() == days
        short_condition = (short_condition.rolling(window=days).sum() == days)

        # Initialize position signals
        position = pd.Series(0, index=data_s.index)

        # Assign positions only at entry points
        position.loc[long_condition] = 1  # Long entry
        position.loc[short_condition] = -1  # Short entry

        # Forward-fill the positions to maintain them until the next change in the condition
        position = position.replace(0, np.nan)
        position = position.ffill().fillna(0)
        position.loc[(long_condition == False) & (short_condition == False)] = 0 # No position if both are false
        position.loc[(long_condition == True) & (short_condition == True)] = 0  # No position if both are true

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
            sl = alfa_short * donchian['high'].shift(1)
            # donchian_short_calc = donchian_short_calc
            lines = {'long': EMVWAP_long_calc, 'short': sl}#donchian_short_calc}
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
    results = backtester.run(donchian_avarage_strategy(short_window=5, long_window=15))

    generate_report_backtest(results['data'])