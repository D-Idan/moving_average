from datetime import datetime

import optuna
import pandas as pd

from backtester.backtester import Backtester
from backtester.performance import generate_report_backtest
from data.data_loader import load_data, preprocess_data
from optuna_opt.loss_func import profit_loss, profit_time_loss, profit_ratio_loss, sharp_ratio_loss
from strategies.emvwap import emvwap_strategy

# Load sample data
# ticker = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA", "AVGO", "NVDA", "META"]
ticker = ["AAPL", "JPM", "SQ", "TSLA", "INTC", "F", "WBD"] # 7 stocks
# ticker = "META"
# ticker = "SPY"
# ticker = "SPMO"
# ticker = "JPM"

start_date = "2000-01-01"
end_date = "2020-01-01"
# end_date = datetime.now().strftime("%Y-%m-%d")

# Number of trials
n_trials = 2000

# Define initial trial parameters
initial_params = [
{'short_window': 15, 'long_window': 223, 'alfa_short': -13, 'alfa_long': 181, 'volume_power_short': 150, 'volume_power_long': 236},
    {'short_window': 1000, 'long_window': 1000, 'alfa_short': 100, 'alfa_long': 100, 'volume_power_short': 100, 'volume_power_long': 100},
    {"short_window": 63, "long_window": 63*4, "alfa_short": 0, "alfa_long": 0, "volume_power_short": 100, "volume_power_long": 100},
    {'short_window': 5, 'long_window': 470, 'alfa_short': 1, 'alfa_long': 3, 'volume_power_short': 160, 'volume_power_long': 47},
    {'short_window': 426, 'long_window': 5, 'alfa_short': 105, 'alfa_long': -14, 'volume_power_short': 166, 'volume_power_long': 71},
    {'short_window': 23, 'long_window': 467, 'alfa_short': -20, 'alfa_long': 141, 'volume_power_short': 133, 'volume_power_long': 223},
]

def loss_flow(strategy, data_pd):

    # Run backtest
    backtester = Backtester(data_pd)
    results = backtester.run(strategy)

    # Calculate total return as the optimization target
    # loss = -profit_loss(results["data"])
    # loss = profit_time_loss(results["data"], w_profit=10**10, w_time=1)
    # loss = sharp_ratio_loss(results["data"])
    loss = profit_ratio_loss(results["data"], w_profit=1, w_time=0.0, w_ratio=0.0, w_entry=0.00)

    return loss

# Function to add initial trials
def add_initial_trials(study, initial_params):
    for params in initial_params:
        study.enqueue_trial(params)
def postprocess_data(ticker, start_date, end_date):
    raw_data = load_data(ticker, start_date, end_date)
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
    short_window = trial.suggest_int("short_window", 5, 63)  # Range for short_window
    long_window = trial.suggest_int("long_window", 63*2, 63*4)  # Range for long_window
    alfa_short = trial.suggest_int("alfa_short", -20, 200)  # Range for alfa_short (percentage)
    alfa_long = trial.suggest_int("alfa_long", -20, 200)  # Range for alfa_long (percentage)
    volume_power_short = trial.suggest_int("volume_power_short", 80, 150)  # Range for volume_power_short
    volume_power_long = trial.suggest_int("volume_power_long", 80, 250)  # Range for volume_power_long

    # Create the strategy with sampled hyperparameters
    strategy = emvwap_strategy(
        short_window=short_window,
        long_window=long_window,
        alfa_short=alfa_short,
        alfa_long=alfa_long,
        volume_power_short=volume_power_short,
        volume_power_long=volume_power_long,
        sides="long",
    )

    # Calculate the loss
    loss = sum(loss_flow(strategy, data_i) for data_i in data)

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
    best_strategy = emvwap_strategy(
        short_window=best_params["short_window"],
        long_window=best_params["long_window"],
        alfa_short=best_params["alfa_short"],
        alfa_long=best_params["alfa_long"],
        volume_power_short=best_params["volume_power_short"],
        volume_power_long=best_params["volume_power_long"],
        sides="long",
    )

    data_back = data[0]

    backtester = Backtester(data_back)
    results = backtester.run(best_strategy)
    generate_report_backtest(results['data'])