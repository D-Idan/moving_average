import numpy as np
import pandas as pd

from backtester.performance import generate_report_backtest


def std_momentum_strategy(
        std_window=20,  # Rolling window for calculating std of percentage change
        profit_multiple=2.0,  # Multiple of entry day's percentage change for take profit
        sides="long",
        next_day_execution=True,
        return_line=False
):
    def strategy(data_s, return_line=return_line, sides=sides, next_day_execution=next_day_execution,
                 std_window=std_window, profit_multiple=profit_multiple):
        """
        Generate buy signals based on positive days with percentage change higher than rolling std,
        with take profit at 2x the entry day's percentage change and stop loss at the entry day's low.
        """
        if not {"Close", "Open", "High", "Low"}.issubset(data_s.columns):
            raise ValueError("Input data must contain 'Close', 'Open', 'High', and 'Low' columns")

        # Calculate daily percentage change
        data_s['daily_pct_change'] = abs(data_s['Close'] - data_s['Open']) / data_s['Open']

        # Calculate rolling standard deviation of percentage change
        data_s['rolling_std'] = data_s['daily_pct_change'].rolling(window=std_window).std()
        data_s['rolling_ema'] = data_s['Open'].ewm(span=std_window).mean()

        # Identify entry conditions
        entry_condition = (data_s['daily_pct_change'] > data_s['rolling_std']) & (data_s['daily_pct_change'] > 0)
        entry_condition = (data_s['daily_pct_change'] > data_s['rolling_std']) & (data_s['Close'] > data_s['rolling_ema'])

        # Calculate take profit and stop loss levels for all potential entries
        data_s['entry_price'] = np.where(entry_condition, data_s['Open'], np.nan)
        data_s['take_profit'] = np.where(
            entry_condition,
            data_s['Close'] * (1 + (data_s['daily_pct_change'] * profit_multiple)),
            np.nan
        )
        data_s['stop_loss'] = np.where(entry_condition, data_s['Low'], np.nan)

        # Initialize signals
        long_signal = pd.Series(np.nan, index=data_s.index)

        # Mark entry points
        long_signal.loc[entry_condition] = 1

        # Forward-fill signals between entry and exit points
        signal_groups = long_signal.cumsum()
        data_s['active_signal'] = signal_groups

        # Calculate running max and min for active positions
        data_s['max_high'] = data_s.groupby('active_signal')['High'].cummax()
        data_s['min_low'] = data_s.groupby('active_signal')['Low'].cummin()

        # Create exit conditions
        take_profit_hit = (data_s['High'] >= data_s['take_profit'].ffill())
        stop_loss_hit = (data_s['Close'] <= data_s['stop_loss'].ffill())
        exit_condition = take_profit_hit | stop_loss_hit

        # Mark exits in the signal
        long_signal.loc[exit_condition] = 0

        # import matplotlib.pyplot as plt
        # data_s[['take_profit',"Close", "stop_loss"]].ffill().plot()
        # plt.scatter(data_s.index, entry_condition * data_s["Close"] , color='red', label='Take Profit')
        # plt.scatter(data_s.index, exit_condition * data_s["Close"] , color='black', label='exit')
        # plt.scatter(data_s.index, long_signal.ffill() * data_s["Close"] , color='green', label='long signal', marker='v')
        # plt.show()

        # Forward fill active signals for persistence until exit
        signal_groups = (long_signal != 0).cumsum()
        long_signal = long_signal.groupby(signal_groups).transform('first')

        # Apply next-day execution logic
        if next_day_execution:
            long_signal = long_signal.shift(1)

        # Short signal (not used in this strategy)
        short_signal = pd.Series(0, index=data_s.index)

        # Drop temporary columns
        take_profit_line = data_s['take_profit'].ffill()
        stop_loss_line = data_s["stop_loss"].ffill()
        data_s.drop(['daily_pct_change', 'rolling_std', 'entry_price', 'take_profit', 'stop_loss',
                     'max_high', 'min_low', 'active_signal'], axis=1, inplace=True)

        if return_line:
            lines = {'long': take_profit_line, 'short': stop_loss_line}
            return long_signal, short_signal, lines

        return long_signal, short_signal

    return strategy

if __name__ == "__main__":
    # Example usage
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
    results = backtester.run(std_momentum_strategy(std_window=20, profit_multiple=1.0))

    # Generate backtest report
    generate_report_backtest(results['data'])