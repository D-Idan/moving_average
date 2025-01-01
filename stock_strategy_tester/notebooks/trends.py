from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from data.data_loader import load_data, preprocess_data

# Initialize PyTrends
pytrends = TrendReq(hl='en-US', tz=360)

# Define the keyword and timeframe
keyword = ["nasdaq"]
timeframe = "today 3-m"  # Last 3 months

# Fetch Google Trends data
pytrends.build_payload(keyword, cat=0, timeframe=timeframe, geo='', gprop='')
trends_data = pytrends.interest_over_time()

if not trends_data.empty:
    trends_data = trends_data.reset_index()

    # Calculate percentage change in interest
    trends_data["Change"] = trends_data[keyword[0]].pct_change() * 100

    # Define thresholds for decision
    increase_threshold = 20  # High increase signals fear, consider "short"
    decrease_threshold = -20  # Decrease signals optimism, consider "long"

    # Determine signals
    def get_position(change):
        if change > increase_threshold:
            return "Short"
        elif change < decrease_threshold:
            return "Long"
        else:
            return "Hold"

    trends_data["Position"] = trends_data["Change"].apply(get_position)

    # Fetch S&P 500 data from Yahoo Finance
    # sp500 = yf.download("^GSPC", start=trends_data["date"].iloc[0], end=trends_data["date"].iloc[-1])
    # sp500.reset_index(inplace=True)

    raw_data = load_data("SPY", trends_data["date"].iloc[0], trends_data["date"].iloc[-1])
    data = preprocess_data(raw_data)
    data.index = pd.to_datetime(data["Date"])
    sp500 = data

    # Merge S&P 500 data with Google Trends data
    sp500["Date"] = pd.to_datetime(sp500["Date"], utc=True)
    trends_data["date"] = pd.to_datetime(trends_data["date"], utc=True)
    sp500.index = sp500["Date"]
    trends_data.index = trends_data["date"]
    # Delete date columns
    del sp500["Date"]
    del trends_data["date"]
    # Reset the index of sp500 to ensure it's single-level
    combined_data = pd.merge(sp500, trends_data, left_index=True, right_index=True, how="inner")
    combined_data["Date"] = combined_data.index
    # Plot the data
    plt.figure(figsize=(14, 7))

    # Plot S&P 500 index
    plt.plot(combined_data["Date"], combined_data["Close"], label="S&P 500", color="blue", alpha=0.7)

    # Plot positions as markers
    for _, row in combined_data.iterrows():
        if row["Position"] == "Long":
            plt.scatter(row["Date"], row["Close"], color="green", label="Long Position", marker="^")
        elif row["Position"] == "Short":
            plt.scatter(row["Date"], row["Close"], color="red", label="Short Position", marker="v")

    # Add labels and legend
    plt.title("S&P 500 Index with Google Trends Positions (Keyword: 'Debt')", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("S&P 500 Close Price", fontsize=12)
    plt.legend(["S&P 500", "Long Position", "Short Position"], loc="upper left")
    plt.grid(alpha=0.3)
    plt.show()

else:
    print("No data retrieved from Google Trends.")