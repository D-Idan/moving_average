import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        """
        Initialize the backtester with data and parameters.
        
        :param data: DataFrame containing stock market data with 'Close' prices.
        :param transaction_cost: Transaction cost per trade (as a fraction).
        """
        self.data = data
        self.transaction_cost = transaction_cost # TODO: Implement transaction costs
        self.positions = 0
        self.trades = []

    def run(self, strategy):
        """
        Run a given trading strategy on the data.

        :param strategy: A function that generates buy/sell columns based on the data. True for positive in each column.
        """
        self.data["Position"] = 0
        signal_long, signal_short = strategy(self.data)
        self.data["Signal_long"] = signal_long
        self.data["Signal_short"] = signal_short

        # Calculate position based on signals
        # Position: 1 for long, -1 for short, 0 for neutral
        self.data["Position"] = self.data["Signal_long"].astype(int) - self.data["Signal_short"].astype(int)

        # Calculate daily returns
        self.data["Daily_Returns"] = self.data["Close"].pct_change()

        # Calculate system returns
        self.data["Sys_Ret"] = (self.data["Position"] * self.data["Daily_Returns"]).cumsum()

        # Calculate system returns long or short only
        self.data["Sys_Ret_long"] = (self.data["Signal_long"] * self.data["Daily_Returns"]).cumsum()
        self.data["Sys_Ret_short"] = (self.data["Signal_short"] * (-1) * self.data["Daily_Returns"]).cumsum()

        print("Backtesting complete.")

        return {"CAGR": self._calculate_cagr(), "Total_Return": self.data["Sys_Ret"].iloc[-1],
                "Max_Drawdown": self._calculate_max_drawdown(), "Time_in_Market": self._calculate_time_in_market(),
                "Long_Return": self.data["Sys_Ret_long"].iloc[-1], "Short_Return": self.data["Sys_Ret_short"].iloc[-1]}

    def report(self):
        """
        Generate a summary report of the backtesting results.
        """
        print("\nBacktesting Report:")
        print(f"Total Return (%): {100 * self.data['Sys_Ret'].iloc[-1]:.2f}")
        print(f"Long Return (%): {100 * self.data['Sys_Ret_long'].iloc[-1]:.2f}")
        print(f"Short Return (%): {100 * self.data['Sys_Ret_short'].iloc[-1]:.2f}")
        print(f"CAGR (Compound Annual Growth Rate) (%): {100 * self._calculate_cagr():.2f}")
        print(f"Max Drawdown: {self._calculate_max_drawdown():.2f}")
        print(f"Time in Market: {self._calculate_time_in_market():.2f}%")

        print("\nTrade Log: TBD")

    def _calculate_time_in_market(self):
        """
        Calculate the percentage of time spent in the market.

        :return: Time in market as a float.
        """
        return 100 * np.count_nonzero(self.data["Position"]) / len(self.data)

    def _calculate_cagr(self):
        """
        Calculate the Compound Annual Growth Rate (CAGR) of the portfolio.

        :return: CAGR as a float.
        """
        return (self.data["Sys_Ret"].iloc[-1] / self.data["Sys_Ret"].iloc[0]) ** (252 / len(self.data)) - 1

    def _calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown of the portfolio.

        :return: Maximum drawdown as a float.
        """
        return np.min(self.data["Sys_Ret"] / np.maximum.accumulate(self.data["Sys_Ret"])) - 1


    def get_results(self):
        """
        Get the data with calculated portfolio values.

        :return: DataFrame with backtesting results.
        """
        return self.data


if __name__ == "__main__":
    # Example usage
    # Load some stock data
    df = pd.DataFrame({
        "Close": [100, 102, 105, 103, 106, 110, 108],
    })

    # Define a simple strategy: buy when the price increases and sell when it decreases
    def simple_strategy(data):
        data["Signal_long"] = data["Close"].diff() > 0
        data["Signal_short"] = data["Close"].diff() < 0
        return data["Signal_long"], data["Signal_short"]

    # Initialize the backtester
    backtester = Backtester(df)

    # Run the backtest
    results = backtester.run(simple_strategy)

    # Generate a report
    backtester.report()
