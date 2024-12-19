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
ticker = "SPMO"
# ticker = "JPM"
start_date = "2005-01-01"
end_date = "2021-01-01"
# end_date = datetime.now().strftime("%Y-%m-%d")
raw_data = load_data(ticker, start_date, end_date)
data = preprocess_data(raw_data)
data.index = pd.to_datetime(data["Date"])

# Define initial trial parameters
initial_params = [
    {"short_window": 1300, "long_window": 23, "alfa_short": 56, "alfa_long": 36, "volume_power_short": 110, "volume_power_long": 93},
    {"short_window": 976, "long_window": 131, "alfa_short": 88, "alfa_long": 30, "volume_power_short": 98, "volume_power_long": 103}
]

# Function to add initial trials
def add_initial_trials(study, initial_params):
    for params in initial_params:
        study.enqueue_trial(params)


# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    short_window = trial.suggest_int("short_window", 5, 1400)  # Range for short_window
    long_window = trial.suggest_int("long_window", 5, 500)  # Range for long_window
    alfa_short = trial.suggest_int("alfa_short", 1, 100)  # Range for alfa_short (percentage)
    alfa_long = trial.suggest_int("alfa_long", 1, 100)  # Range for alfa_long (percentage)
    volume_power_short = trial.suggest_int("volume_power_short", 90, 110)  # Range for volume_power_short
    volume_power_long = trial.suggest_int("volume_power_long", 90, 110)  # Range for volume_power_long

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
    # loss = profit_loss(results["data"])
    # # loss = profit_time_loss(results["data"], w_profit=0.95, w_time=0.05)
    loss = profit_ratio_loss(results["data"], w_profit=0.89, w_time=0.05, w_ratio=0.05, w_entry=0.01)

    return loss

if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Add initial trials
    add_initial_trials(study, initial_params)

    # Optimize the study
    study.optimize(objective, n_trials=4000)

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
        sides="long",
    )

    backtester = Backtester(data)
    results = backtester.run(best_strategy)
    generate_report_backtest(results['data'])