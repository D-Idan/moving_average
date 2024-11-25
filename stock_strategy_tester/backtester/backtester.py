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
        # Calculate daily returns
        self.data["Daily_Returns"] = self.data["Close"].pct_change()
        self.data["Benchmark_Ret"] = self.data["Daily_Returns"].cumsum()

    def run(self, strategy):
        """
        Run a given trading strategy on the data.

        :param strategy: A function that generates buy/sell columns based on the data. True for positive in each column.
        """
        self.data["Position"] = 0

        data_position = self.data.copy()
        signal_long, signal_short = strategy(data_position)

        data_position["Signal_long"] = signal_long.dropna()
        data_position["Signal_short"] = signal_short.dropna()

        data_position = data_position.dropna()
        data_position["Benchmark_Ret"] = data_position["Daily_Returns"].cumsum()

        # Calculate position based on signals
        # Position: 1 for long, -1 for short, 0 for neutral
        data_position["Position"] = data_position["Signal_long"].astype(int) - data_position["Signal_short"].astype(int)

        # Calculate system returns
        data_position["Sys_Ret"] = (data_position["Position"] * data_position["Daily_Returns"]).cumsum()

        # Calculate system returns long or short only
        data_position["Sys_Ret_long"] = (data_position["Signal_long"] * data_position["Daily_Returns"]).cumsum()
        data_position["Sys_Ret_short"] = (data_position["Signal_short"] * (-1) * data_position["Daily_Returns"]).cumsum()

        return {"data": data_position.copy(), "Total_Return": data_position["Sys_Ret"].iloc[-1],
                "Time_in_Market": self._calculate_time_in_market(),
                "Long_Return": data_position["Sys_Ret_long"].iloc[-1], "Short_Return": data_position["Sys_Ret_short"].iloc[-1],
                "Benchmark_Ret": data_position["Benchmark_Ret"].iloc[-1]}


    def _calculate_time_in_market(self):
        """
        Calculate the percentage of time spent in the market.

        :return: Time in market as a float.
        """
        return 100 * np.count_nonzero(self.data["Position"]) / len(self.data)



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
