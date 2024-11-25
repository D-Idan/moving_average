

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


def plot_benchmark_with_positions(data, ax1=None, save=False, title="Benchmark with Strategy Positions", ax_return=False):
    """
    Plot the benchmark (stock itself) graph with the position of the strategy.

    :param data: DataFrame containing the backtesting results with 'Close' and 'Position' columns.
    """
    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        own_fig = True  # Indicates this function created its own figure
    else:
        own_fig = False

    ax1.plot(data.index, data['Close'], label='Benchmark (Close Price)', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data['Position'], label='Position', color='red', alpha=0.3)
    ax2.set_ylabel('Position', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    if ax_return:
        ax1.set_title(title)
        return ax1

    plt.title('Benchmark with Strategy Positions')
    plt.legend(loc='upper left')

    if own_fig:
        fig.tight_layout()
        if save:
            path_results = Path(f"./results/{title}.jpg")
            plt.savefig(path_results)
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

def plot_gains(data, ax=None, save=False, title="Benchmark vs Strategy Gains", ax_return=False):
    """
    Plot the benchmark gains and the strategy gains.
    :param data: DataFrame containing the backtesting results with 'Benchmark_Ret' and 'Sys_Ret' columns.
    :param ax: Existing Axes object for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    data_benchmark = data["Daily_Returns"].cumsum()
    data_strategy = (data["Position"] * data["Daily_Returns"]).cumsum()
    ax.plot(data.index, data_benchmark, label='Benchmark Gains', color='blue')
    ax.plot(data.index, data_strategy, label='Strategy Gains', color='green')
    ax.set_ylabel('Cumulative Returns')
    ax.set_title(title)
    ax.legend()

    if ax_return:
        return ax

    if save:
        path_results = Path(f"./results/{title}.jpg")
        plt.savefig(path_results)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def plot_drawdown(data, ax=None, save=False, title="Strategy Drawdown", ax_return=False):
    """
    Plot the drawdown of the strategy.
    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :param ax: Existing Axes object for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    cumulative_returns = (data["Position"] * data["Daily_Returns"]).cumsum()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    ax.plot(data.index, drawdown, label='Drawdown', color='red')
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.legend()

    if ax_return:
        return ax
    if save:
        path_results = Path(f"./results/{title}.jpg")
        plt.savefig(path_results)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def plot_volatility(data, ax=None, save=False, title="Strategy Rolling Volatility", ax_return=False):
    """
    Plot the rolling volatility of the strategy and benchmark.
    :param data: DataFrame containing the backtesting results with 'Sys_Ret' and 'Benchmark_Ret' columns.
    :param ax: Existing Axes object for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate rolling volatility
    data_sys_ret = data["Position"] * data["Daily_Returns"]
    strategy_volatility = data_sys_ret.rolling(window=21).std() * (252 ** 0.5)
    benchmark_volatility = data['Daily_Returns'].rolling(window=21).std() * (252 ** 0.5)

    # Plot rolling volatilities
    ax.plot(data.index, strategy_volatility, label='Strategy Volatility (21 days)', color='purple')
    ax.plot(data.index, benchmark_volatility, label='Benchmark Volatility (21 days)', color='orange', linestyle='--')
    ax.set_ylabel('Volatility')
    ax.set_title(title)
    ax.legend()

    if ax_return:
        return ax
    if save:
        path_results = Path(f"./results/{title}.jpg")
        plt.savefig(path_results)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_yearly_comparison(yearly_results, save=False, title="Yearly Strategy vs Benchmark Returns"):
    """
    Plot a column chart comparing strategy results (total, long, and short) with benchmark results for each year.

    :param yearly_results: List of dictionaries containing yearly results.
    :param save: Boolean indicating whether to save the plot as a file.
    :param title: Title of the chart.
    """
    # Extract data
    years = [result["year"] for result in yearly_results]
    strategy_returns = [result["strategy_return"] for result in yearly_results]
    strategy_long_returns = [result["long_return"] for result in yearly_results]
    strategy_short_returns = [result["short_return"] for result in yearly_results]
    benchmark_returns = [result["benchmark_return"] for result in yearly_results]

    # Setup x positions
    x = range(len(years))
    bar_width = 0.2  # Adjust bar width for better clarity

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the bars
    ax.bar(x, strategy_returns, bar_width, label='Total Strategy Return', color='blue')
    ax.bar([p + bar_width for p in x], strategy_long_returns, bar_width, label='Strategy Long Return', color='green')
    ax.bar([p + 2 * bar_width for p in x], strategy_short_returns, bar_width, label='Strategy Short Return', color='red')
    ax.bar([p + 3 * bar_width for p in x], benchmark_returns, bar_width, label='Benchmark Return', color='gray')

    # Customize the plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Return')
    ax.set_title(title)
    ax.set_xticks([p + 1.5 * bar_width for p in x])  # Center the labels
    ax.set_xticklabels(years)
    ax.legend()

    # Save or show the plot
    if save:
        path_results = Path(f"./results/{title}.jpg")
        plt.savefig(path_results)
    else:
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