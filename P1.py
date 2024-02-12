import streamlit as st
import datetime as dt
import yfinance as yf
from yahooquery import Ticker
import pandas as pd
import numpy as np

def main():
    st.title("Portfolio Analysis")

    start = st.date_input("Enter the Starting Point for the Historical Data (Preferable 5 years or Less)", value=dt.datetime(2020, 1, 1), max_value=dt.datetime.now(), format="DD.MM.YYYY",)

    # Calculate start and end dates
    end_date = dt.datetime.now()
    start_date = start

    st.write(f"Start Date: {start_date}")
    st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")

    # Multi-select option for stock tickers
    df = pd.read_csv('tickers.csv')
    selected_tickers = st.multiselect("Select your stocks", options = df.Symbol)
    
    st.title('Yahoo Finance Data Statistics')

    # Fetch financial data
    ticker = Ticker(selected_tickers) 
    fin_data_dict = ticker.financial_data
    fin_data_df = pd.DataFrame.from_dict(fin_data_dict, orient='index').T

    # Display data
    st.subheader(f"Financial Data for {selected_tickers}")
    st.write(fin_data_df)

    # Fetch historical price data for the selected stocks using yfinance
    try:
        portfolio = yf.download(selected_tickers, start=start_date, end=end_date)

        # Extract the closing prices for each selected stock
        portfolio = portfolio['Close']

        # Ensure the index is in datetime format
        portfolio.index = pd.to_datetime(portfolio.index, format='mixed', infer_datetime_format=True)

        portfolio_returns = portfolio.pct_change().dropna()

        # User input for stock weights
        st.write("The Sum of the Weights must be Equal to One - Else they are Equally Divided")
        weights = st.text_input("Enter Weights for each Stock (Comma-Separated)", "")
        
        weights = [float(w.strip()) for w in weights.split(",")] if weights else None

        # Handle case for a single stock
        if weights is None or len(weights) != len(selected_tickers) or sum(weights) != 1:
            weights = [1/len(selected_tickers)] * len(selected_tickers)

        # Calculate portfolio return
        if len(selected_tickers) == 1:
            portfolio_returns_1 = portfolio_returns
        else:
            portfolio_returns_1 = portfolio_returns.dot(weights)
            
        
        portfolio_returns_1_df = pd.DataFrame({"Your Portfolio": portfolio_returns_1})
        
        # Fetch historical price data for the S&P 500
        
        sp = yf.download('^GSPC', start=start_date, end=end_date)

        # Extract the closing prices for each selected stock
        sp = sp['Close']

        # Ensure the index is in datetime format
        sp.index = pd.to_datetime(sp.index, format='mixed', infer_datetime_format=True)

        sp_returns = sp.pct_change().dropna()
        sp_returns_df = pd.DataFrame({"S&P 500": sp_returns})
        
        # Combining Portfolio with S&P 500
        combined_df = pd.concat([portfolio_returns_1_df, sp_returns_df], axis="columns", join="inner")

        # Quantitative Analysis Results
        st.write("## Quantitative Analysis")

        # Plot daily returns of the portfolio
        st.subheader("Daily Returns for Portfolio")
        st.line_chart(combined_df)

        # Calculate cumulative returns of the portfolio
        cumulative_returns = (1 + combined_df).cumprod() - 1

        # Plot cumulative returns for the portfolio
        st.subheader("Cumulative Returns for Portfolio")
        st.line_chart(cumulative_returns)

        # Calculate the daily standard deviation of the portfolio
        std_deviation = combined_df.std()

        # Calculate the annualized standard deviation (252 trading days)
        annualized_std_deviation = std_deviation * np.sqrt(252)
        st.subheader("Annualized Standard Deviation")
        st.write(pd.DataFrame(annualized_std_deviation, columns=["SD"]))

        # Calculate rolling standard deviation
        ewm_std = combined_df.ewm(halflife=21).std()
        st.subheader("21-Day Rolling Exponentially Weighted Standard Deviation")
        st.line_chart(ewm_std)

        # Calculate the correlation
        portfolio_correlation = combined_df.corr().loc["Your Portfolio", "S&P 500"]

        st.subheader("Correlation between Your Portfolio and S&P 500")
        st.write(f"{portfolio_correlation:.2f}")

        # Calculate variance of S&P 500
        rolling_variance = combined_df["S&P 500"].rolling(window=60).var()

        # Calculate rolling beta values for each portfolio
        rolling_beta = (combined_df["Your Portfolio"].rolling(window=60).cov(combined_df["S&P 500"])) / rolling_variance

        st.subheader("60-Day Rolling Beta for Your Portfolio vs S&P 500")
        st.line_chart(rolling_beta)
        
        # Annualized Sharpe Ratios (252 trading days)
        sharpe_ratios = (combined_df.mean() * 252) / annualized_std_deviation
        st.subheader("Annualized Sharpe Ratio")
        st.write(pd.DataFrame(sharpe_ratios, columns=["Value"]))

    except Exception as e:
        st.write(f"Something isn't Right!: {e}")

if __name__ == "__main__":
    main()
