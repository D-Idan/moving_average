# Technology
ticker_tech = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "AMD", "ADBE", "CRM"]

# Consumer Discretionary
ticker_consumer_discretionary = ["AMZN", "TSLA", "NKE", "HD", "MCD", "SBUX", "TGT", "LVS", "BKNG", "CMG"]

# Energy
ticker_energy = ["XOM", "CVX", "COP", "OXY", "SLB", "HAL", "EOG", "PXD", "VLO", "PSX"]

# Financials
ticker_financials = ["JPM", "BAC", "WFC", "C", "GS", "MS", "BK", "BLK", "AXP", "SCHW"]

# Healthcare
ticker_healthcare = ["UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "TMO", "ABT", "BMY", "CVS"]

# Industrials
ticker_industrials = ["BA", "CAT", "RTX", "GE", "DE", "HON", "LMT", "NOC", "GD", "MMM"]

# Consumer Staples
ticker_consumer_staples = ["PG", "KO", "PEP", "WMT", "COST", "MO", "PM", "CL", "MDLZ", "KMB"]

# Utilities
ticker_utilities = ["NEE", "DUK", "SO", "D", "AEP", "XEL", "PEG", "EXC", "SRE", "ED"]

# Communication Services
ticker_communication_services = ["T", "VZ", "TMUS", "DIS", "NFLX", "CHTR", "GOOGL", "META", "CMCSA", "ATVI"]

# Materials
ticker_materials = ["LIN", "APD", "ECL", "NEM", "FCX", "DOW", "CTVA", "IFF", "ALB", "MLM"]

# Real Estate
ticker_real_estate = ["PLD", "AMT", "CCI", "EQIX", "SPG", "O", "SBAC", "DLR", "WY", "VTR"]

if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd

    # Define moving average crossover strategies
    ma_strategies = {
        "low_volatility": {"short_window": 5, "long_window": 20},
        "medium_volatility": {"short_window": 10, "long_window": 50},
        "high_volatility": {"short_window": 20, "long_window": 100},
    }

    # Group tickers by sector
    sector_tickers = {
        "Technology": ticker_tech,
        "Consumer Discretionary": ticker_consumer_discretionary,
        "Energy": ticker_energy,
        "Financials": ticker_financials,
        "Healthcare": ticker_healthcare,
        "Industrials": ticker_industrials,
        "Consumer Staples": ticker_consumer_staples,
        "Utilities": ticker_utilities,
        "Communication Services": ticker_communication_services,
        "Materials": ticker_materials,
        "Real Estate": ticker_real_estate,
    }


    # Helper function to fetch stock data and calculate beta and volatility
    def calculate_metrics(ticker):
        try:
            data = yf.download(ticker, period="1y", interval="1d")
            data["Returns"] = data["Adj Close"].pct_change()
            volatility = data["Returns"].std()
            beta = yf.Ticker(ticker).info.get("beta", 1)  # Default beta = 1 if unavailable
            return {"ticker": ticker, "volatility": volatility, "beta": beta}
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None


    # Analyze each sector and group tickers
    grouped_tickers = {"low_volatility": [], "medium_volatility": [], "high_volatility": []}

    for sector, tickers in sector_tickers.items():
        print(f"Processing sector: {sector}")
        metrics = [calculate_metrics(ticker) for ticker in tickers]
        metrics = [m for m in metrics if m]  # Filter out None values

        for metric in metrics:
            if metric["volatility"] <= 0.02 and metric["beta"] < 1:
                grouped_tickers["low_volatility"].append(metric["ticker"])
            elif 0.02 < metric["volatility"] <= 0.04 and 1 <= metric["beta"] <= 1.5:
                grouped_tickers["medium_volatility"].append(metric["ticker"])
            else:
                grouped_tickers["high_volatility"].append(metric["ticker"])

    # Output groups and corresponding strategies
    for group, tickers in grouped_tickers.items():
        print(
            f"\n{group.capitalize()} group ({ma_strategies[group]['short_window']}-{ma_strategies[group]['long_window']} MA strategy):")
        print(", ".join(tickers))