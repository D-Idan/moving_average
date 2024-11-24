

import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import seaborn as sns
from pathlib import Path

def plot_heatmap(data, periods_fast, periods_slow, center_value=0, title="Strategy Performance Heatmap",
                 save=False, legend=False):
    """
    Plot a heatmap of the strategy performance for different moving average periods.

    :param data: DataFrame containing the performance results.
    :param periods_fast: List of fast moving average periods.
    :param periods_slow: List of slow moving average periods.
    :param center_value: Value at which to center the colormap.
    """

    plt.figure(figsize=(16, 8))
    # Fill NaN values with 0
    data = data.fillna(0).astype(float)
    # Use DivergingNorm for enhanced visual contrast around the center value
    sns.heatmap(data, cmap="PiYG", linewidth=0.5, linecolor="black",
                xticklabels=periods_slow, yticklabels=periods_fast, center=center_value,
                annot=legend, fmt=".2f")
    plt.title(title)
    plt.xlabel("Slow Moving Average Period")
    plt.ylabel("Fast Moving Average Period")
    if save:
        path_results = Path(f"./results/{title}.jpg")
        plt.savefig(path_results)
    else:
        plt.show()


def plot_aggregate_heatmap(data, bin_size, center_value=0, title="Strategy Performance Heatmap",
                           bests_print=False, save=False, legend=True):
    """
    Aggregate the heatmap data by combining multiple cells into larger bins.

    :param data: pandas DataFrame containing the heatmap data.
    :param bin_size: Integer specifying the size of the bins (e.g., 4, 16, 32).
    :param center_value: Value around which the heatmap is centered.
    :param title: Title of the heatmap plot.
    :param bests_print: Boolean to determine if the top 5 maximum values should be printed.
    :return: Aggregated DataFrame.
    """
    # Ensure the dimensions of the DataFrame are divisible by the bin size
    rows, cols = data.shape
    aggregated_data = data.iloc[:rows - (rows % bin_size), :cols - (cols % bin_size)]

    # Group rows and columns into bins and calculate the average
    aggregated_data = aggregated_data.groupby(
        aggregated_data.index // bin_size).mean().T.groupby(
        aggregated_data.columns // bin_size).mean().T

    plot_heatmap(aggregated_data,
                 periods_fast=aggregated_data.index * bin_size,
                 periods_slow=aggregated_data.columns * bin_size,
                 center_value=center_value,
                 title=title,
                 save=save,
                 legend=legend)

    # Print the top 5 maximum values and their positions
    if bests_print:
        top_5 = aggregated_data.stack().astype(float).nlargest(5) # Get the 5 largest values with their indices
        print("\nTop 5 maximum values in the aggregated heatmap:")
        for (x, y), value in top_5.items():
            print(f"Fast Period: {x * bin_size}, Slow Period: {y * bin_size}, Value: {value}")




    # if bests_print:
    #     bests = aggregated_data.dropna().stack().nlargest(5)
    #     print(f"Best 5 results: {bests}")


def plot_benchmark_with_positions(data):
    """
    Plot the benchmark (stock itself) graph with the position of the strategy.

    :param data: DataFrame containing the backtesting results with 'Close' and 'Position' columns.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(data.index, data['Close'], label='Benchmark (Close Price)', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data['Position'], label='Position', color='red', alpha=0.3)
    ax2.set_ylabel('Position', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    plt.title('Benchmark with Strategy Positions')
    plt.legend(loc='upper left')
    plt.show()

def plot_gains(data):
    """
    Plot the benchmark gains and the strategy gains.

    :param data: DataFrame containing the backtesting results with 'Benchmark_Ret' and 'Sys_Ret' columns.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Benchmark_Ret'], label='Benchmark Gains', color='blue')
    plt.plot(data.index, data['Sys_Ret'], label='Strategy Gains', color='green')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Benchmark Gains vs Strategy Gains')
    plt.legend()
    plt.show()

def plot_drawdown(data):
    """
    Plot the drawdown of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    """
    cumulative_returns = data['Sys_Ret'].cumsum()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, drawdown, label='Drawdown', color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Strategy Drawdown')
    plt.legend()
    plt.show()

def plot_volatility(data):
    """
    Plot the rolling volatility of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    """
    rolling_volatility = data['Sys_Ret'].rolling(window=21).std() * (252 ** 0.5)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, rolling_volatility, label='Rolling Volatility (21 days)', color='purple')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('Strategy Rolling Volatility')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    from backtester import Backtester
    from strategies.moving_average import moving_average_strategy

    # Load sample data
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    data = yf.download(ticker, start=start_date, end=end_date)

    # Initialize backtester
    backtester = Backtester(data)
    # Run the backtest
    results = backtester.run(moving_average_strategy(short_window=5, long_window=15))

    # Plot the results
    plot_benchmark_with_positions(results["data"])
    plot_gains(results["data"])
    plot_drawdown(results["data"])
    plot_volatility(results["data"])