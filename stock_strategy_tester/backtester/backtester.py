import pandas as pd


class Backtester:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        """
        Initialize the backtester with data and parameters.
        
        :param data: DataFrame containing stock market data with 'Close' prices.
        :param initial_balance: Initial capital for the portfolio.
        :param transaction_cost: Transaction cost per trade (as a fraction).
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.balance = initial_balance
        self.positions = 0
        self.portfolio_value = initial_balance
        self.trades = []

    def run(self, strategy):
        """
        Run a given trading strategy on the data.

        :param strategy: A function that generates buy/sell signals.
        """
        self.data["Signal"] = strategy(self.data)
        # Calculate cumulative position based on signals
        self.data["Position"] = self.data["Signal"].cumsum()
        # Ensure no negative positions
        self.data["Position"] = self.data["Position"].clip(lower=0)

        for i in range(len(self.data)):
            self._execute_trade(i)

        self.data["Portfolio Value"] = self.portfolio_value
        print("Backtesting complete.")

    def _execute_trade(self, index):
        """
        Execute trades based on the current signal.

        :param index: Index of the current row in the data.
        """
        row = self.data.iloc[index]
        signal = row["Signal"]
        close_price = row["Close"]

        if signal > 0:  # Buy signal
            self._buy(close_price, signal)
        elif signal < 0:  # Sell signal
            self._sell(close_price, abs(signal))

        # Update portfolio value
        self.portfolio_value = self.balance + self.positions * close_price

    def _buy(self, price, quantity):
        """
        Simulate buying stocks.

        :param price: Price of the stock.
        :param quantity: Number of shares to buy.
        """
        cost = quantity * price * (1 + self.transaction_cost)
        if self.balance >= cost:
            self.balance -= cost
            self.positions += quantity
            self.trades.append(
                {"Type": "Buy", "Quantity": quantity, "Price": price})
        else:
            print("Not enough balance to buy.")

    def _sell(self, price, quantity):
        """
        Simulate selling stocks.

        :param price: Price of the stock.
        :param quantity: Number of shares to sell.
        """
        if self.positions >= quantity:
            revenue = quantity * price * (1 - self.transaction_cost)
            self.balance += revenue
            self.positions -= quantity
            self.trades.append(
                {"Type": "Sell", "Quantity": quantity, "Price": price})
        else:
            print("Not enough positions to sell.")

    def report(self):
        """
        Generate a summary report of the backtesting results.
        """
        total_return = self.portfolio_value - self.initial_balance
        roi = (total_return / self.initial_balance) * 100
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Total Return: ${total_return:,.2f}")
        print(f"Return on Investment (ROI): {roi:.2f}%")

        trade_log = pd.DataFrame(self.trades)
        print("\nTrade Log:")
        print(trade_log)

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
        return data["Close"].diff().fillna(0).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # Initialize and run backtester
    backtester = Backtester(df)
    backtester.run(simple_strategy)
    backtester.report()
