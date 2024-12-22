from datetime import datetime

import optuna
import pandas as pd

from backtester.backtester import Backtester
from backtester.performance import generate_report_backtest
from data.data_loader import load_data, preprocess_data
from optuna_opt.loss_func import profit_loss, profit_time_loss, profit_ratio_loss
from strategies.emvwap import emvwap_strategy

# Load sample data
# ticker = "META"
# ticker = "SPMO"
ticker = "JPM"
start_date = "2005-01-01"
end_date = "2021-01-01"
# end_date = datetime.now().strftime("%Y-%m-%d")
raw_data = load_data(ticker, start_date, end_date)
data = preprocess_data(raw_data)
data.index = pd.to_datetime(data["Date"])

# Define initial trial parameters
initial_params = [
    {'short_window': 1000, 'long_window': 1000, 'alfa_short': 100, 'alfa_long': 100, 'volume_power_short': 100, 'volume_power_long': 100},
    {"short_window": 63, "long_window": 63*4, "alfa_short": 0, "alfa_long": 0, "volume_power_short": 100, "volume_power_long": 100},
    {'short_window': 5, 'long_window': 470, 'alfa_short': 1, 'alfa_long': 3, 'volume_power_short': 160, 'volume_power_long': 47},
    {'short_window': 426, 'long_window': 5, 'alfa_short': 105, 'alfa_long': -14, 'volume_power_short': 166, 'volume_power_long': 71},
    {'short_window': 23, 'long_window': 467, 'alfa_short': -20, 'alfa_long': 141, 'volume_power_short': 133, 'volume_power_long': 223},
]

# Function to add initial trials
def add_initial_trials(study, initial_params):
    for params in initial_params:
        study.enqueue_trial(params)


# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    short_window = trial.suggest_int("short_window", 10, 400)  # Range for short_window
    long_window = trial.suggest_int("long_window", 10, 500)  # Range for long_window
    alfa_short = trial.suggest_int("alfa_short", -20, 120)  # Range for alfa_short (percentage)
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

    # Run backtest
    backtester = Backtester(data)
    results = backtester.run(strategy)

    # Calculate total return as the optimization target
    # loss = -profit_loss(results["data"])
    # loss = profit_time_loss(results["data"], w_profit=0.95, w_time=0.05)
    loss = profit_ratio_loss(results["data"], w_profit=1.00, w_time=0.0, w_ratio=0.0, w_entry=0.00)

    return loss

if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Add initial trials
    add_initial_trials(study, initial_params)

    # Optimize the study
    study.optimize(objective, n_trials=2000)

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

    backtester = Backtester(data)
    results = backtester.run(best_strategy)
    generate_report_backtest(results['data'])