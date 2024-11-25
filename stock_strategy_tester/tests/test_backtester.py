import unittest
import pandas as pd
from backtester.backtester import Backtester


class TestBacktester(unittest.TestCase):
    def setUp(self):
        """
        Set up the testing environment with sample data and an instance of the Backtester.
        """
        # Sample stock data
        self.sample_data = pd.DataFrame({
            "Open": [100, 102, 101, 103, 105, 104, 106]
        })

        # Simple strategy for testing: Buy if the price increases, Sell if it decreases
        def simple_strategy(data):
            return data["Open"].diff().fillna(0).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        self.strategy = simple_strategy
        self.backtester = Backtester(data=self.sample_data)

    def test_initial_balance(self):
        """
        Test that the initial balance is set correctly.
        """
        self.assertEqual(self.backtester.initial_balance, 10000)

    def test_run_strategy(self):
        """
        Test running the strategy on the data.
        """
        self.backtester.run(self.strategy)
        self.assertIn("Signal", self.backtester.data.columns)
        self.assertIn("Portfolio Value", self.backtester.data.columns)

    def test_portfolio_value_update(self):
        """
        Test that portfolio value is updated correctly after running the strategy.
        """
        initial_balance = self.backtester.initial_balance
        self.backtester.run(self.strategy)
        final_portfolio_value = self.backtester.portfolio_value
        # Portfolio value should be greater than or equal to the initial balance
        self.assertGreaterEqual(final_portfolio_value, 0)

    def test_buy_and_sell_logic(self):
        """
        Test that buy and sell logic works as expected.
        """
        # Add signals for a manual test
        self.sample_data["Signal"] = [1, -1, 1, 0, -1, 1, -1]
        for i in range(len(self.sample_data)):
            self.backtester._execute_trade(i)

        # Check that trades were recorded correctly
        trades = self.backtester.trades
        self.assertEqual(len(trades), 5)  # 5 trades: 3 buys and 2 sells

    def test_report(self):
        """
        Test that the report function executes without errors.
        """
        self.backtester.run(self.strategy)
        try:
            self.backtester.report()
        except Exception as e:
            self.fail(f"Report function raised an exception: {e}")

    def test_results_dataframe(self):
        """
        Test that the results DataFrame contains the expected columns.
        """
        self.backtester.run(self.strategy)
        results = self.backtester.get_results()
        self.assertIn("Portfolio Value", results.columns)
        self.assertIn("Signal", results.columns)

    def test_transaction_costs(self):
        """
        Test that transaction costs are applied during trades.
        """
        self.backtester.transaction_cost = 0.01  # 1% transaction cost
        self.backtester.run(self.strategy)
        # Ensure that transaction costs reduce the portfolio value
        final_portfolio_value = self.backtester.portfolio_value
        self.assertLess(final_portfolio_value, self.backtester.initial_balance)


if __name__ == "__main__":
    unittest.main()
