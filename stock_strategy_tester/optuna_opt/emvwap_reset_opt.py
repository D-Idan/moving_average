from datetime import datetime

import optuna
import pandas as pd
from sympy.plotting.intervalmath import interval

from backtester.backtester import Backtester
from backtester.performance import generate_report_backtest
from data.data_loader import load_data, preprocess_data
from data import tickers_by_sector
from optuna_opt.loss_func import profit_loss, profit_time_loss, profit_ratio_loss, sharp_ratio_loss
from strategies.donchian_avarage import donchian_avarage_strategy
from strategies.emvwap import emvwap_strategy
from strategies.emvwap_reset import emvwap_strategy_with_reset

# Load sample data
# ticker = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA", "AVGO", "NVDA", "META"]
# ticker = ["AAPL", "JPM", "SQ", "TSLA", "INTC", "F", "WBD"] # 7 stocks
ticker = tickers_by_sector.ticker_financials
# ticker = tickers_by_sector.ticker_real_estate + ticker
# ticker = tickers_by_sector.ticker_materials + ticker
# ticker = tickers_by_sector.ticker_industrials + ticker
# ticker = tickers_by_sector.ticker_consumer_staples + ticker
# ticker = tickers_by_sector.ticker_consumer_discretionary + ticker
# ticker = "META"
# ticker = "TSLA"
# ticker = "SPY"
# ticker = "SPMO"
# ticker = "JPM"
# ticker = "C"
# ticker = "F"
ticker = "AAPL"
# ticker = ["AAPL", "MSFT", "GOOGL", "AMZN"]
# ticker = ["AAPL"]
# ticker = ["cost"]
# ticker = "USDGBP=X"

data_interval = "1d"
# data_interval = "5d"
# data_interval = "30m"
start_date = "2000-01-01"
end_date = "2020-01-01"
# end_date = datetime.now().strftime("%Y-%m-%d")

# Strategy
# strategy_selected = emvwap_strategy
strategy_selected = emvwap_strategy_with_reset

# Number of trials
n_trials = 2000
sides = "short"

# Define initial trial parameters
initial_params = [
# {'short_window': 101, 'long_window': 168, 'alfa_short': 40, 'alfa_long': 0, 'volume_power_short': 110, 'volume_power_long': 50, 'long_diff': 8, 'short_diff': 80},# 5D
# {'short_window': 202, 'long_window': 256, 'alfa_short': 90, 'alfa_long': 30, 'volume_power_short': 180, 'volume_power_long': 90, 'long_diff': 16}, # DONCHIAN
#     {"short_window": 63, "long_window": 63*4, "alfa_short": 0, "alfa_long": 0, "volume_power_short": 100, "volume_power_long": 100},
#     {'short_window': 5, 'long_window': 470, 'alfa_short': 1, 'alfa_long': 3, 'volume_power_short': 160, 'volume_power_long': 47},
#     {'short_window': 5, 'long_window': 25, 'alfa_short': 108, 'alfa_long': 137, 'volume_power_short': 88, 'volume_power_long': 89},
#     {'short_window': 23, 'long_window': 467, 'alfa_short': -20, 'alfa_long': 141, 'volume_power_short': 133, 'volume_power_long': 223},

{'short_window': 63, 'long_window': 63*2, 'volume_power_short': 100, 'volume_power_long': 100, 'long_diff': 5, 'reset_window': 5, 'confirm_days': 1},
{'short_window': 61, 'long_window': 128, 'volume_power_short': 140, 'volume_power_long': 100, 'long_diff': 64, 'reset_window': 18, 'confirm_days': 2}, # APPL BOTH value: -0.56
]

def loss_flow(strategy, data_pd):

    # Run backtest
    backtester = Backtester(data_pd)
    results = backtester.run(strategy)

    # Calculate total return as the optimization target
    loss = -profit_loss(results["data"], all_positions=False, normalize=True, add_cumulative=True)
    # loss = profit_time_loss(results["data"], w_profit=1, w_time=1)
    # loss = sharp_ratio_loss(results["data"])
    # loss = profit_ratio_loss(results["data"], w_profit=0.0, w_time=0.0, w_ratio=0.9, w_entry=0.0)

    return loss

# Function to add initial trials
def add_initial_trials(study, initial_params):
    for params in initial_params:
        study.enqueue_trial(params)
def postprocess_data(ticker, start_date, end_date):
    raw_data = load_data(ticker, start_date, end_date, interval=data_interval)
    data_p = preprocess_data(raw_data)
    data_p.index = pd.to_datetime(data_p["Date"])
    return data_p

def load_data_for_testing(ticker, start_date, end_date):
    if type(ticker) == str:
        ticker = [ticker]

    data_list = []
    for t in ticker:
        data_s = postprocess_data(t, start_date, end_date)
        data_s = data_s.dropna()
        data_list.append(data_s)
    return data_list

data = load_data_for_testing(ticker, start_date, end_date)



# Objective function for Optuna
def objective(trial):

    # Define hyperparameters to optimize
    short_window = trial.suggest_int("short_window", 5, 61, step=7)  # Range for short_window
    long_window = trial.suggest_int("long_window", 64, 64*4, step=32)  # Range for long_window
    volume_power_short = trial.suggest_int("volume_power_short", 80, 180, step=20)  # Range for volume_power_short # DONCHIAN
    volume_power_long = trial.suggest_int("volume_power_long", 80, 160, step=20)  # Range for volume_power_long
    long_diff = trial.suggest_int("long_diff", 0, 64, step=8)
    reset_window = trial.suggest_int("reset_window", 2, 20, step=2)
    confirm_days = trial.suggest_int("confirm_days", 2, 4, step=1)

    # Create the strategy with sampled hyperparameters
    strategy = strategy_selected(
        short_window=short_window,
        long_window=long_window,
        volume_power_short=volume_power_short,
        volume_power_long=volume_power_long,
        long_diff=long_diff,
        reset_window=reset_window,
        confirm_days=confirm_days,
        sides=sides,
    )

    # Calculate the loss
    loss = sum(loss_flow(strategy, data_i) for data_i in data) / len(data)

    return loss

if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Add initial trials
    add_initial_trials(study, initial_params)

    # Optimize the study
    study.optimize(objective, n_trials=n_trials)

    # Print the best hyperparameters
    print("\nBest hyperparameters:")
    print(study.best_params)

    # Optional: Run the strategy with the best hyperparameters and generate a report
    best_params = study.best_params
    # best_strategy = emvwap_strategy(
    best_strategy = strategy_selected(
        short_window=best_params["short_window"],
        long_window=best_params["long_window"],
        volume_power_short=best_params["volume_power_short"], # DONCHIAN
        volume_power_long=best_params["volume_power_long"],
        long_diff=best_params["long_diff"],
        reset_window=best_params["reset_window"],
        confirm_days=best_params["confirm_days"],
        sides=sides,
    )

    for i, data_i in enumerate(data):
        print(f"\n Backtesting for {ticker[i]}")
        backtester = Backtester(data_i)
        results = backtester.run(best_strategy)
        generate_report_backtest(results['data'])
