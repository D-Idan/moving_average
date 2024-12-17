import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.plotter import plot_heatmap, plot_aggregate_heatmap, plot_benchmark_with_positions, plot_gains, \
    plot_drawdown, plot_volatility, plot_yearly_comparison


def calculate_sharpe_ratio(data, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :param risk_free_rate: Risk-free rate for Sharpe Ratio calculation.
    :return: Sharpe Ratio as a float.
    """
    excess_returns = data - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(data):
    """
    Calculate the maximum drawdown of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :return: Maximum drawdown as a float.
    """
    cumulative_returns = data.cumsum()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_cagr(data):
    """
    Calculate the Compound Annual Growth Rate (CAGR) of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :return: CAGR as a float.
    """
    total_return = data.iloc[-1]
    n_years = len(data) / 252  # Assuming 252 trading days in a year
    return (1 + total_return) ** (1 / n_years) - 1

def calculate_volatility(data):
    """
    Calculate the annualized volatility of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :return: Annualized volatility as a float.
    """
    return np.std(data) * np.sqrt(252)

def calculate_sortino_ratio(data, risk_free_rate=0.0):
    """
    Calculate the Sortino Ratio of the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :param risk_free_rate: Risk-free rate for Sortino Ratio calculation.
    :return: Sortino Ratio as a float.
    """
    excess_returns = data - risk_free_rate
    downside_risk = np.std(excess_returns[excess_returns < 0])
    return np.mean(excess_returns) / downside_risk

def calculate_performance_metrics(data, risk_free_rate=0.0):
    """
    Calculate various performance metrics for the strategy.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :param risk_free_rate: Risk-free rate for performance metrics calculation.
    :return: Dictionary containing performance metrics.
    """
    return {
        "Total Return": data["Sys_Ret"].iloc[-1],
        "Long Return": data["Sys_Ret_long"].iloc[-1],
        "Short Return": data["Sys_Ret_short"].iloc[-1],
        "Time in Market": 100 * np.count_nonzero(data["Position"]) / len(data),

        "Sharpe Ratio": calculate_sharpe_ratio(data, risk_free_rate),
        "Max Drawdown": calculate_max_drawdown(data["Sys_Ret"]),
        "CAGR": calculate_cagr(data["Sys_Ret"]),
        "Volatility": calculate_volatility(data["Sys_Ret"]),
        "Sortino Ratio": calculate_sortino_ratio(data["Sys_Ret"], risk_free_rate)
    }

def generate_report_benchmark(data, risk_free_rate=0.0):
    """
    Generate a summary report of the benchmark results.

    :param data: DataFrame containing the backtesting results with 'Benchmark_Ret' column.
    :param risk_free_rate: Risk-free rate for performance metrics calculation.
    """
    data_bench_Ret = data["Daily_Returns"].cumsum()
    print("\nBenchmark Report:")
    print(f"Total Return (%): {100 * data_bench_Ret.iloc[-1]:.2f}")
    print(f"CAGR (Compound Annual Growth Rate) (%): {100 * calculate_cagr(data_bench_Ret):.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(data_bench_Ret):.2f}")
    print(f"Time in Market: 100.00%")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(data_bench_Ret, risk_free_rate):.2f}")
    print(f"Volatility: {calculate_volatility(data_bench_Ret):.2f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(data_bench_Ret, risk_free_rate):.2f}")

def generate_report_backtest(data, risk_free_rate=0.0):
    """
    Generate a summary report of the backtesting results.
    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :param risk_free_rate: Risk-free rate for performance metrics calculation.
    """
    data_Sys_Ret = (data["Position"] * data["Daily_Returns"]).cumsum()
    print("\nBacktesting Report:")
    print(f"Total Return (%): {100 * (data["Position"] * data["Daily_Returns"]).sum():.2f}")
    print(f"Long Return (%): {100 * (data["Signal_long"] * data["Daily_Returns"]).sum():.2f}")
    print(f"Short Return (%): {-100 * (data["Signal_short"] * data["Daily_Returns"]).sum():.2f}")
    print(f"CAGR (Compound Annual Growth Rate) (%): {100 * calculate_cagr(data_Sys_Ret):.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(data_Sys_Ret):.2f}")
    print(f"Time in Market: {100 * np.count_nonzero(data['Position']) / len(data):.2f}%")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(data_Sys_Ret, risk_free_rate):.2f}")
    print(f"Volatility: {calculate_volatility(data_Sys_Ret):.2f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(data_Sys_Ret, risk_free_rate):.2f}")

    print("\nTrade Log: ")
    print(f"Number of changes in Position: {data['Position'].diff().abs().sum()}")

def generate_report(data, risk_free_rate=0.0):
    """
    Generate a summary report of the backtesting results.

    :param data: DataFrame containing the backtesting results with 'Sys_Ret' column.
    :param risk_free_rate: Risk-free rate for performance metrics calculation.
    """
    generate_report_benchmark(data, risk_free_rate=0.0)

    generate_report_backtest(data, risk_free_rate=0.0)

def run_oneSETvalues_backtest(backtester, config, strategy_tested):
    """
    Run a backtest, store results, and generate reports.

    :param backtester: Instance of the Backtester class.
    :param config: Configuration object containing initial balance, transaction cost, and risk-free rate.
    :param strategy_tested: Strategy function to test.
    """
    backtest_results = backtester.run(strategy_tested)
    backtest_results_data = backtest_results["data"]

    unique_years = backtest_results_data['Date'].dt.year.unique()
    yearly_results = []

    for year in unique_years:
        print(f"\n\nResults for {year} \n")
        yearly_backtest_results = backtest_results_data[backtest_results_data['Date'].dt.year == year]

        # Generate a report
        generate_report(yearly_backtest_results, risk_free_rate=config.RISK_FREE_RATE)

        # Plot the results
        plot_benchmark_with_positions(yearly_backtest_results, title=f"Backtest Results for {year}", save=True)
        plot_gains(yearly_backtest_results, title=f"Backtest gains for {year}", save=True)
        plot_drawdown(yearly_backtest_results, title=f"Backtest Drawdown for {year}", save=True)
        plot_volatility(yearly_backtest_results, title=f"Backtest Volatility for {year}", save=True)

        # Store yearly results
        yearly_results.append({
            "year": year,
            "strategy_return": (yearly_backtest_results["Position"] * yearly_backtest_results["Daily_Returns"]).sum(),
            "benchmark_return": yearly_backtest_results["Daily_Returns"].sum(),
            "long_return": (yearly_backtest_results["Signal_long"] * yearly_backtest_results["Daily_Returns"]).sum(),
            "short_return": (yearly_backtest_results["Signal_short"] * -yearly_backtest_results["Daily_Returns"]).sum(),
        })

    # Plot yearly comparison
    plot_yearly_comparison(yearly_results, title="Yearly Comparison", save=True)

def run_Nvalues_backtest(data, periods_fast, periods_slow, backtester, config, strategy_tested, bin_size=16):
    """
    Run backtests for each combination of moving averages, store results, and generate reports.

    :param data: DataFrame containing the stock data.
    :param periods_fast: Range of periods for the short moving average.
    :param periods_slow: Range of periods for the long moving average.
    :param backtester: Instance of the Backtester class.
    :param config: Configuration object containing initial balance, transaction cost, and risk-free rate.
    """
    # Get the unique years in the data
    unique_years = data['Date'].dt.year.unique()
    results = pd.DataFrame(index=periods_fast, columns=periods_slow)
    yearly_results = {year: pd.DataFrame(index=periods_fast, columns=periods_slow) for year in unique_years}

    # Test each combination of moving averages
    for short_window in tqdm(periods_fast):
        for long_window in periods_slow:
            strategy = strategy_tested(short_window, long_window)
            backtest_results = backtester.run(strategy)

            # Store the results
            backtest_results_data = backtest_results["data"]
            backtest_results_sum = (backtest_results_data["Position"] * backtest_results_data["Daily_Returns"]).sum()
            results.loc[short_window, long_window] = backtest_results_sum

            for year in unique_years:
                yearly_backtest_results = backtest_results_data[backtest_results_data['Date'].dt.year == year]
                yearly_backtest_results_sum = (yearly_backtest_results["Position"] * yearly_backtest_results["Daily_Returns"]).sum()
                yearly_results[year].loc[short_window, long_window] = yearly_backtest_results_sum

    # Generate a report
    generate_report_benchmark(backtest_results["data"], risk_free_rate=config.RISK_FREE_RATE)

    # Plot the heatmap
    plot_heatmap(results, periods_fast, periods_slow, center_value=backtest_results["Benchmark_Ret"],
                 title="Full Backtest Results", save=True)
    plot_aggregate_heatmap(results, bin_size=bin_size, center_value=backtest_results["Benchmark_Ret"],
                           title="Full Backtest Results", bests_print=True, save=True)

    for year in unique_years:
        print(f"\n\nResults for {year} \n")
        yearly_benchmark = data[data['Date'].dt.year == year]
        yearly_benchmark_sum = yearly_benchmark["Daily_Returns"].sum()
        print(f"Yearly Benchmark: {yearly_benchmark_sum}")

        plot_heatmap(yearly_results[year], periods_fast, periods_slow, center_value=yearly_benchmark_sum,
                     title=f"Backtest Results for {year}", save=True)
        plot_aggregate_heatmap(yearly_results[year], bin_size=bin_size, center_value=yearly_benchmark_sum,
                               title=f"Backtest Results for {year}", bests_print=True, save=True)




if __name__ == "__main__":
    # Example usage
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
    # Generate a report
    backtester.report()

    # Generate report
    performance_metrics = generate_report(results["data"])

