import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
# Assuming PyPortfolioOpt is installed and used for optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns


# Placeholder function for loading data
def load_data():
    # Replace with actual data loading logic
    # For demonstration, creating a dummy DataFrame with random data
    dates = pd.date_range(start="2010-01-01", end="2020-12-31", freq="M")
    data = pd.DataFrame(np.random.rand(len(dates), 4), index=dates,
                        columns=['Asset 1', 'Asset 2', 'Asset 3', 'Asset 4'])
    return data


# Placeholder for the optimization logic

def optimize_portfolio(data, gamma):
    # Calculate expected returns and the sample covariance matrix
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Calculate performance metrics including Sharpe ratio
    perf = ef.portfolio_performance(verbose=False)
    sharpe_ratio = perf[2]  # Assuming perf returns (return, volatility, sharpe_ratio)

    return cleaned_weights, sharpe_ratio


# Main app
import streamlit as st
import plotly.express as px


def main():
    st.title("Dynamic Portfolio Optimization Tool")


    # User inputs for the period sizes
    in_sample_years = st.number_input('In-sample period size (years)', min_value=1, max_value=10, value=5)
    out_of_sample_years = st.number_input('Out-of-sample period size (years)', min_value=1, max_value=5, value=2)

    gamma = st.slider('Risk Aversion Coefficient', min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # Load data (this should be replaced with actual data loading logic)
    data = load_data()

    # Display the loaded data (optional)
    if st.checkbox('Show raw data'):
        st.write(data)

    # Split the data based on user input
    split_date = data.index.max() - pd.DateOffset(years=out_of_sample_years)
    in_sample_data = data[:split_date]
    out_of_sample_data = data[split_date:]

    #optimized_weights, sharpe_ratio = optimize_portfolio(in_sample_data, gamma)

    chart_type = st.radio(
        "Select Chart Type for Weights Visualization:",
        ('Line Chart', 'Bar Chart')
    )

    if st.button('Optimize Portfolio'):
        if not in_sample_data.empty and not out_of_sample_data.empty:
            optimized_weights, sharpe_ratio = optimize_portfolio(in_sample_data, gamma)
            # Display the Sharpe ratio
            st.write(f'Sharpe Ratio: {sharpe_ratio}')
            st.write('Optimized Portfolio Weights:', optimized_weights)
            # Display the optimized weights with selected chart type
            weights_df = pd.DataFrame(list(optimized_weights.items()), columns=['Asset', 'Weight'])
            if chart_type == 'Bar Chart':
                fig = px.bar(weights_df, x='Asset', y='Weight', title="Optimized Portfolio Weights")
            else:
                # Line chart with markers for weights
                fig = px.line(weights_df, x='Asset', y='Weight', title="Optimized Portfolio Weights", markers=True)
            st.plotly_chart(fig)
        else:
            st.error("Insufficient data for optimization. Please check your dataset and period sizes.")


if __name__ == "__main__":
    main()

