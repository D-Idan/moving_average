import numpy as np

from backtester.performance import calculate_sharpe_ratio, calculate_total_profit_per_trade


def profit_loss(data, all_positions=False, normalize=False, add_cumulative=False):
    """
    Custom loss function to maximize profit.
    :param data: DataFrame containing backtesting results.
    :return: Loss value (lower is better).
    """
    relevant_returns = data["Position"] * data["Daily_Returns"]
    # Calculate profit (total returns)
    if all_positions:
        relevant_returns = data["Daily_Returns"]
        return data["Close"].iloc[-1] / data["Close"].iloc[0]

    total_profit = calculate_total_profit_per_trade(data, normalize=normalize)

    # total_profit = (relevant_returns + 1).prod()
    cumulative_profit = relevant_returns.sum() / data["Daily_Returns"].sum() if normalize else relevant_returns.sum()

    if add_cumulative:
        return (total_profit + cumulative_profit) / 2
    return total_profit

def profit_time_loss(data, w_profit=0.7, w_time=0.3):
    """
    Custom loss function to balance profit and time in market.
    :param data: DataFrame containing backtesting results.
    :param w_profit: Weight for the profit term (higher is better).
    :param w_time: Weight for the time in market penalty (lower is better).
    :return: Loss value (lower is better).
    """
    # Calculate time in market as a percentage
    time_in_market = np.count_nonzero(data["Position"]) / len(data)

    # Normalize metrics (optional, but ensures consistency across scales)
    # Assuming total_profit is on a larger scale, we normalize it.
    profit_score = profit_loss(data, normalize=True, add_cumulative=True) # Normalize profit score
    time_penalty = time_in_market + 1.0  # This is already normalized (0 to 1)

    # Calculate loss
    loss = -1 * (w_profit * profit_score) / (w_time * time_penalty) # We need to minimize the loss

    return loss


def profit_ratio_loss(data, w_profit=0.5, w_time=0.3, w_ratio=0.2, w_entry=0.1):
    """
    Custom loss function to maximize profitable trade ratio, profit, and minimize time in the market.
    :param data: DataFrame containing backtesting results.
    :param w_profit: Weight for the profit term (higher is better).
    :param w_time: Weight for the time in market penalty (lower is better).
    :param w_ratio: Weight for the profitable trades ratio (higher is better).
    :param w_entry: Weight for the entry to long position (lower is better).
    :return: Loss value (lower is better).
    """

    position = data["long_signal"] - data["short_signal"]  # 1 for long, -1 for short, 0 for none

    # Identify profitable and unprofitable trades for the entire active period
    data["Position"] = position
    data["Trade_ID"] = (position != position.shift(1)).cumsum()  # Assign unique IDs to each trade period


    # Calculate time in market as a percentage
    time_in_market = np.count_nonzero(data["Position"]) / len(data)

    # Calculate cumulative returns for each trade
    data["Cumulative_Returns"] = data.groupby("Trade_ID")["Daily_Returns"].cumsum()

    # Determine if a trade is profitable or not
    trade_results = data.groupby("Trade_ID").agg(
        {"Position": "first", "Cumulative_Returns": "last", "Open": "first"}
    )

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
    if num_unprofitable_trades + num_profitable_trades > 0:
        profitable_ratio = num_profitable_trades / (num_unprofitable_trades + num_profitable_trades)
    else:
        profitable_ratio = -10**8  # Infinite ratio if no unprofitable trades

    # Calculate profit (total returns)
    profit_score = profit_loss(data, normalize=True, add_cumulative=True)  # Normalize profit score

    # Normalize metrics
    time_penalty = time_in_market + 1  # This is already normalized (0 to 1)

    # short_signal touch lows for entrance to long position
    entry_long = (abs(data["short_signal"] - data["Low"]) / (data["Low"])).sum()

    # Calculate loss We need to minimize the loss
    loss = -(w_profit * profit_score / w_time * time_penalty) - w_ratio * profitable_ratio - w_entry * entry_long

    return loss

def sharp_ratio_loss(data):
    """
    Custom loss function to maximize sharp ratio
    :return: Loss value (lower is better).
    """
    relevant_returns = data["Position"] * data["Daily_Returns"]
    data_Sys_Ret = (relevant_returns).cumsum()
    sr = calculate_sharpe_ratio(data_Sys_Ret)

    # Calculate loss We need to minimize the loss
    loss = -sr

    return loss