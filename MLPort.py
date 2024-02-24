import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Placeholder for ML forecasting
def ml_forecast(tickers):
    # Simulate expected returns and covariance matrix
    expected_returns = pd.Series(np.random.rand(len(tickers)), index=tickers)
    cov_matrix = pd.DataFrame(np.random.rand(len(tickers), len(tickers)), index=tickers, columns=tickers)
    return expected_returns, cov_matrix

# Classical forecasting using historical data
def classical_forecast(tickers):
    data = yf.download(tickers, start="2020-01-01", end="2021-01-01")['Adj Close']
    expected_returns = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)
    return expected_returns, cov_matrix

def optimize_portfolio(expected_returns, cov_matrix):

    ef = EfficientFrontier(expected_returns, cov_matrix)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return cleaned_weights

def main():
    st.title("Portfolio Optimization with Forecasting Choice")

    # User choices
    method = st.radio("Forecasting Method", ["Machine Learning", "Classical"])
    available_tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
    selected_tickers = st.multiselect("Select Tickers", available_tickers, default=["AAPL", "GOOG"])

    if st.button("Optimize Portfolio"):
        if not selected_tickers:
            st.error("Please select at least one ticker.")
            return

        if method == "Machine Learning":
            exp_returns, cov_matrix = ml_forecast(selected_tickers)
        else:  # Classical
            exp_returns, cov_matrix = classical_forecast(selected_tickers)

        optimized_weights = optimize_portfolio(exp_returns, cov_matrix)
        st.write("Optimized Portfolio Weights:")
        st.json(optimized_weights)

if __name__ == "__main__":
    main()


from pypfopt import risk_models, expected_returns, plotting