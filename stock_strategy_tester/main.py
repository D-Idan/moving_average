# main.py
from datetime import datetime

import pandas as pd
import yfinance as yf
from backtester import Backtester
from strategies.arima import arma_strategy
from strategies.emvwap import emvwap_strategy
from strategies.exponential_moving_average import exponential_moving_average_strategy
from strategies.ichimoku_cloud_strategy import ichimoku_cloud_strategy
from strategies.mean_reversion import mean_reversion_strategy
from strategies.moving_average import moving_average_strategy
from strategies.moving_ptcv_average import ptcv_strategy
from strategies.moving_vwap import moving_vwap_strategy
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
    # ticker = "META"
    # ticker = "SPY"
    # ticker = "SHOP"
    # ticker = "BRK-B"
    ticker = "JPM"
    # ticker = "BAC"
    # ticker = "VYM"
    # ticker = "LUMI.TA"
    # ticker = "BEZQ.TA"
    # ticker = "TSLA"
    start_date = "2017-01-01"
    # Todays date
    end_date = datetime.now().strftime("%Y-%m-%d")
    # Load and preprocess data
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Initialize backtester
    backtester = Backtester(data, initial_balance=config.INITIAL_BALANCE, transaction_cost=config.TRANSACTION_COST)

    # # Run the backtest for one combination
    # strategy = moving_average_strategy(short_window=15, long_window=150)
    # strategy = moving_vwap_strategy(short_window=15, long_window=150)
    # strategy = exponential_moving_average_strategy(short_window=10, long_window=13)
    # strategy = ichimoku_cloud_strategy(short_window=50, long_window=70) #50/70
    # strategy = mean_reversion_strategy( window=116, num_std_dev=3, sides="both")
    # strategy = ptcv_strategy(short_window=80, long_window=100, sides="both")
    params = {'short_window': 426, 'long_window': 5, 'alfa_short': 105, 'alfa_long': -14, 'volume_power_short': 166, 'volume_power_long': 71}
    params = {'short_window': 28, 'long_window': 322, 'alfa_short': -11, 'alfa_long': 117, 'volume_power_short': 97, 'volume_power_long': 147}

    strategy = emvwap_strategy(**params)

    run_oneSETvalues_backtest(backtester, config, strategy_tested=strategy)

    # 144 / 112 | 15 / 5


    ########################################################## Multiple values ##########################################################
    # # # # Define the range of moving average periods to test
    periods_fast = range(5, 300, 10)
    periods_slow = range(5, 300, 10)
    # periods_slow = [x / 10 for x in range(1, 40, 1)]
    #
    # strategy = moving_average_strategy
    # strategy = exponential_moving_average_strategy
    # strategy = ichimoku_cloud_strategy
    # strategy = mean_reversion_strategy
    # strategy = arma_strategy
    # strategy = ptcv_strategy
    # strategy = moving_vwap_strategy
    strategy = emvwap_strategy
    # #
    # # # Run the backtest for each combination of moving average periods - Yearly
    # run_Nvalues_backtest(data, periods_fast, periods_slow, backtester, config,
    #                                          strategy_tested=strategy, bin_size=10)



if __name__ == "__main__":
    main()