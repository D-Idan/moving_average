from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.performance import generate_report_backtest
from strategies.emvwap import emvwap_strategy

def plot_strategy_results(data, lines, results):
    """
    Plot the results of the EMVWAP strategy.
    :param data: The DataFrame with strategy signals and price data.
    :param lines: A dictionary containing the EMVWAP long and short lines.
    :param results: The backtest results containing trades and performance.
    """
    price = data["Close"]
    long_line = lines["long"]
    short_line = lines["short"]
    position = data["long_signal"] - data["short_signal"]  # 1 for long, -1 for short, 0 for none

    # Identify profitable and unprofitable trades for the entire active period
    data["Position"] = position
    data["Trade_ID"] = (position != position.shift(1)).cumsum()  # Assign unique IDs to each trade period

    # Calculate cumulative returns for each trade
    data["Cumulative_Returns"] = data.groupby("Trade_ID")["Daily_Returns"].cumsum()

    # Determine if a trade is profitable or not
    trade_results = data.groupby("Trade_ID").agg(
        {"Position": "first", "Cumulative_Returns": "last", "Open": "first"}
    )
    trade_results["Profitable"] = trade_results["Cumulative_Returns"] > 0

    # Broadcast profitability back to the original DataFrame
    data["Profitable"] = data["Trade_ID"].map(trade_results["Profitable"])

    # Mark profitable and unprofitable trades
    profitable_trades = data[data["Profitable"] & (data["Position"] != 0)]
    unprofitable_trades = data[~data["Profitable"] & (data["Position"] != 0)]

    # Calculate the number of profitable and unprofitable trades
    num_profitable_trades = len(profitable_trades["Trade_ID"].unique())
    num_unprofitable_trades = len(unprofitable_trades["Trade_ID"].unique())

    # Calculate the ratio of profitable to unprofitable trades
    if num_unprofitable_trades > 0:
        profitable_ratio = num_profitable_trades / num_unprofitable_trades
    else:
        profitable_ratio = float("inf")  # Infinite ratio if no unprofitable trades

    # Print the results
    print(f"Number of Profitable Trades: {num_profitable_trades}")
    print(f"Number of Unprofitable Trades: {num_unprofitable_trades}")
    print(f"Profitable vs Unprofitable Trades Ratio: {profitable_ratio:.2f}")

    plt.figure(figsize=(15, 8))

    # Plot price and EMVWAP lines
    plt.plot(price, label="Price", color="blue", alpha=0.7)
    plt.plot(long_line, label="EMVWAP Long", color="green", linestyle="--")
    plt.plot(short_line, label="EMVWAP Short", color="red", linestyle="--")

    # Highlight long and short positions
    plt.fill_between(
        data.index, price.min(), price.max(),
        where=(position == 1), color="green", alpha=0.2, label="Long Position"
    )
    plt.fill_between(
        data.index, price.min(), price.max(),
        where=(position == -1), color="red", alpha=0.2, label="Short Position"
    )

    # Highlight profitable and unprofitable trades for the entire active period
    plt.scatter(profitable_trades.index, profitable_trades["Open"], color="lime", label="Profitable Trade", marker="^", alpha=0.8)
    # plt.scatter(unprofitable_trades.index, unprofitable_trades["Open"], color="crimson", label="Unprofitable Trade", marker="v", alpha=0.8)

    # Add labels, legend, and title
    plt.title(f"EMVWAP Strategy Performance")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Load data and preprocess (as per your script)
    from backtester.backtester import Backtester
    from data.data_loader import load_data, preprocess_data

    # Load sample data
    # ticker = "META"
    ticker = "JPM"
    start_date = "2010-01-01"
    # Today date
    # end_date = "2021-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)
    data.index = pd.to_datetime(data["Date"])

    # Strategy
    params = {'short_window': 42, 'long_window': 306, 'alfa_short': 17, 'alfa_long': 65}
    strategy = emvwap_strategy(**params)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the strategy with return_line=True to get the EMVWAP lines
    long_signal, short_signal, lines = strategy(data, return_line=True)

    # Add signals to the data
    data["long_signal"] = long_signal
    data["short_signal"] = short_signal

    # Run the backtest
    results = backtester.run(strategy)
    generate_report_backtest(results['data'])

    # Plot the results
    plot_strategy_results(data, lines, results)