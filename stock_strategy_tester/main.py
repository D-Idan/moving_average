# main.py
import pandas as pd
import yfinance as yf
from backtester import Backtester
from strategies.moving_average import moving_average_strategy
from utils.plotter import plot_benchmark_with_positions, plot_gains, plot_drawdown, plot_volatility, plot_heatmap, \
    plot_aggregate_heatmap
from utils.plot_strategies import plot_moving_averages
from backtester.performance import generate_report, generate_report_benchmark, run_Nvalues_backtest, \
    run_oneSETvalues_backtest
import config
from data.data_loader import load_data, preprocess_data
import tqdm

def main():
    # Load sample data
    # ticker = "AAPL"
    # ticker = "SPY"
    # ticker = "SHOP"
    # ticker = "BRK-B"
    ticker = "JPM"
    # ticker = "LUMI.TA"
    # ticker = "BEZQ.TA"
    # ticker = "TSLA"
    start_date = "2017-01-01"
    end_date = "2024-01-01"
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Initialize backtester
    backtester = Backtester(data, initial_balance=config.INITIAL_BALANCE, transaction_cost=config.TRANSACTION_COST)

    # # Run the backtest for one combination
    # strategy = moving_average_strategy(short_window=5, long_window=20)
    # run_oneSETvalues_backtest(backtester, config, strategy_tested=strategy)

    # 144 / 112 | 15 / 5


    ########################################################## Multiple values ##########################################################
    # # # Define the range of moving average periods to test
    periods_fast = range(1, 50, 1)
    periods_slow = range(1, 300, 5)
    strategy = moving_average_strategy

    # Run the backtest for each combination of moving average periods - Yearly
    run_Nvalues_backtest(data, periods_fast, periods_slow, backtester, config,
                                             strategy_tested=strategy, bin_size=10)



if __name__ == "__main__":
    main()