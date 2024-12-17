import optuna
from backtester.backtester import Backtester
from backtester.performance import generate_report_backtest
from data.data_loader import load_data, preprocess_data
from strategies.emvwap import emvwap_strategy


# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    short_window = trial.suggest_int("short_window", 5, 300)  # Range for short_window
    long_window = trial.suggest_int("long_window", 5, 400)  # Range for long_window
    alfa_short = trial.suggest_int("alfa_short", 10, 200)  # Range for alfa_short (percentage)
    alfa_long = trial.suggest_int("alfa_long", 10, 200)  # Range for alfa_long (percentage)

    # Create the strategy with sampled hyperparameters
    strategy = emvwap_strategy(
        short_window=short_window,
        long_window=long_window,
        alfa_short=alfa_short,
        alfa_long=alfa_long,
        sides="long",
    )

    # Load data for backtesting
    ticker = "JPM"
    start_date = "2017-01-01"
    end_date = "2021-01-01"
    raw_data = load_data(ticker, start_date, end_date)
    data = preprocess_data(raw_data)

    # Run backtest
    backtester = Backtester(data)
    results = backtester.run(strategy)

    # Calculate total return as the optimization target
    total_return = (results['data']["Position"] * results['data']["Daily_Returns"]).sum()

    return total_return

if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

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
    raw_data = load_data("JPM", "2017-01-01", "2021-01-01")
    data = preprocess_data(raw_data)
    backtester = Backtester(data)
    results = backtester.run(best_strategy)
    generate_report_backtest(results['data'])