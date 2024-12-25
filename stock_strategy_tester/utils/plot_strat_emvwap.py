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
        {"Position": "first", "Cumulative_Returns": "last", "Open": "first", "Close": "last"}
    )
################################ 1 ################################
    # Calculate percentage profit for each trade
    trade_results["Profit_Percentage"] = trade_results.apply(
        lambda row: ((row["Close"] - row["Open"]) / row["Open"] * 100)
        if row["Position"] == 1 else ((row["Open"] - row["Close"]) / row["Open"] * 100),
        axis=1
    )

    # delete position 0 trades
    trade_results1 = trade_results[trade_results["Position"] != 0]

    # Broadcast profit percentages back to the original DataFrame
    data["Profit_Percentage"] = data["Trade_ID"].map(trade_results1["Profit_Percentage"])

    # Print profit percentages for each trade
    u = 1
    qqq = trade_results1[["Profit_Percentage"]]
    for i in range(len(qqq)):
        u *= ((qqq.iloc[i] / 100) + 1)

    u = u.iloc[0] * 100 - 100 if len(qqq) > 0 else 1
    print(f"Compound Profit Percentage: {u:.2f}%")
    ################################ 1 ################################

    # Determine if a trade is profitable or not
    trade_results["Profitable"] = (
            (trade_results["Position"] == 1) & (trade_results["Cumulative_Returns"] > 0) |  # Longs: Positive returns
            (trade_results["Position"] == -1) & (trade_results["Cumulative_Returns"] < 0)  # Shorts: Negative returns
    )

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
    # Annotate trades with profit percentages
    for idx, trade in trade_results.iterrows():
        trade_data = data[data["Trade_ID"] == idx]
        if not trade_data.empty:
            trade_date = trade_data.index[0]  # Pick a representative date
            if trade["Position"] != 0:
                plt.text(
                    trade_date, trade["Open"], f"{trade['Profit_Percentage']:.2f}%",
                    color="black" if trade["Profit_Percentage"] > 0 else "red",
                    fontsize=10, ha="center", alpha=0.7
                )

    # Highlight entry and exit points
    for idx, trade in trade_results.iterrows():
        trade_data = data[data["Trade_ID"] == idx]
        if not trade_data.empty and trade["Position"] != 0:
            entry_date = trade_data.index[0]  # Entry point (first date of trade)
            exit_date = trade_data.index[-1]  # Exit point (last date of trade)

            # Draw vertical lines for entry and exit points
            plt.axvline(x=entry_date, color='green', linestyle='-', alpha=0.5, label="Entry Point" if idx == 0 else "")
            plt.axvline(x=exit_date, color='red', linestyle='-', alpha=0.5, label="Exit Point" if idx == 0 else "")

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
    # ticker = "TSLA"
    # ticker = "F"
    # ticker = "SHOP"
    # ticker = "SQ"
    # ticker = "LVS"
    # ticker = "CMG"
    # ticker = "SPMO"
    # ticker = "spy"
    ticker = "U"
    # ticker = "JPM"
    # ticker = "AMD"
    # ticker = "nvda"
    # ticker = "AXP"
    # ticker = "SBAC"

    start_date = "2020-01-01"
    # start_date = "2000-01-01"
    # Today date
    # end_date = "2021-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)
    data.index = pd.to_datetime(data["Date"])

    # Strategy
    # params = {'short_window': 64, 'long_window': 64*4, 'alfa_short': 0, 'alfa_long': 0, 'volume_power_short': 100, 'volume_power_long': 100}
    # params = {'short_window': 5, 'long_window': 64, 'alfa_short': 100, 'alfa_long': 100, 'volume_power_short': 100, 'volume_power_long': 100}
    # params = {'short_window': 5, 'long_window': 64*2, 'alfa_short': 100, 'alfa_long': 100, 'volume_power_short': 100, 'volume_power_long': 100}
    # params = {'short_window': 10, 'long_window': 64*2, 'alfa_short': 0, 'alfa_long': 0, 'volume_power_short': 150, 'volume_power_long': 100} # SPY short above price
    # params = {'short_window': 3, 'long_window': 52, 'alfa_short': 0, 'alfa_long': 0, 'volume_power_short': 184, 'volume_power_long': 120} # JPM short above price
    params = {'short_window': 25, 'long_window': 98, 'alfa_short': 10, 'alfa_long': 0, 'volume_power_short': 162, 'volume_power_long': 92}

    # params = {'short_window': 6, 'long_window': 219, 'alfa_short': 0, 'alfa_long': 0, 'volume_power_short': 130, 'volume_power_long': 98}
    # params = {'short_window': 6, 'long_window': 217, 'alfa_short': 0, 'alfa_long': 0, 'volume_power_short': 185, 'volume_power_long': 97}

    # params = {'short_window': 27, 'long_window': 59, 'alfa_short': 47, 'alfa_long': 97, 'volume_power_short': 111, 'volume_power_long': 118} # TSLA
    # params = {'short_window': 19, 'long_window': 5, 'alfa_short': 52, 'alfa_long': 40, 'volume_power_short': 124, 'volume_power_long': 101} # F / TSLA
    # params = {'short_window': 22, 'long_window': 59, 'alfa_short': 14, 'alfa_long': 49, 'volume_power_short': 101, 'volume_power_long': 112} # FCX
    params['next_day_execution'] = True
    params['sides'] = "long"

    strategy = emvwap_strategy(**params)


    # Initialize backtester
    backtester = Backtester(data)
    # Run the strategy with return_line=True to get the EMVWAP lines
    long_signal, short_signal, lines = strategy(data, return_line=True)

    # Add signals to the data
    data["long_signal"] = long_signal
    data["short_signal"] = short_signal

    # Shift signals back

    # Run the backtest
    results = backtester.run(strategy)
    generate_report_backtest(results['data'])

    # Plot the results
    plot_strategy_results(data, lines, results)