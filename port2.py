import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Download historical data
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')
    data['Returns'] = data['Adj Close'].pct_change()
    return data


#classical

def es_forecast(data):
    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    er_forecast_ex = model_fit.forecast()[0]
    var_forecast_ex = np.var(data)  # Simplification
    return er_forecast_ex, var_forecast_ex

def arima_forecast(data):
    model_returns = ARIMA(data, order=(1, 1, 1))
    model_fit_returns = model_returns.fit()
    er_forecast_ar = model_fit_returns.forecast()[0]
    residuals_squared = model_fit_returns.resid**2
    model_variance = ARIMA(residuals_squared, order=(1, 1, 1))
    model_fit_variance = model_variance.fit()
    var_forecast_ar = model_fit_variance.forecast()[0]
    return er_forecast_ar, var_forecast_ar


# Historical forecasting method
def historical_forecast(data):
    er_forecast = np.mean(data)
    var_forecast = np.var(data)
    return er_forecast, var_forecast


# Machine learning forecasting method


def ml_forecast(data):
    # Convert datetime index to a numerical format (e.g., integer sequence)
    # This assumes `data` is a pandas Series with a datetime index
    X = np.arange(len(data)).reshape(-1, 1)[:-1]  # Use a simple integer sequence as features
    y = data.values[1:]  # Shift data to predict the next step

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1e3, gamma=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'XGBoost': XGBRegressor(n_estimators=100),
        'LightGBM': LGBMRegressor(n_estimators=100)
    }

    best_model_name = None
    best_model_score = np.inf

    for name, model in models.items():
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_val)
        score = mean_squared_error(y_val, predictions)

        if score < best_model_score:
            best_model_name = name
            best_model_score = score
            best_model = model

    # Use the best model to forecast the next value
    er_forecast_ml = best_model.predict([[len(data)]]).item()
    var_forecast_ml = best_model_score

    return er_forecast_ml, var_forecast_ml, best_model_name


# Calculation of rolling optimal weights and evaluate performance metrics
def calculate_and_evaluate(data, gamma, in_sample_years, forecast_method):
    monthly_optimal_weights = []
    monthly_returns = []
    for end_date in data.index[in_sample_years * 12:]:
        in_sample_data = data[:end_date].tail(in_sample_years * 12)
        excess_returns = in_sample_data['Returns'] - in_sample_data['Monthly Yield']

        if forecast_method == 'Classical:ARIMA':
            er_forecast_ar, var_forecast_ar = arima_forecast(excess_returns)
            optimal_weight = (1 / gamma) * (er_forecast_ar / var_forecast_ar)
            monthly_optimal_weights.append(optimal_weight)

            # Assuming reinvestment in the index with the calculated optimal weight
            next_month_return = data['Returns'][end_date] * optimal_weight
            monthly_returns.append(next_month_return)
            # Performance Metrics
            cumulative_return = np.prod(np.array(monthly_returns) + 1) - 1
            mean_return = np.mean(monthly_returns)
            std_return = np.std(monthly_returns)
            sharpe_ratio = mean_return / std_return * np.sqrt(12)  # Annualized Sharpe Ratio
        elif forecast_method == 'Classical:Exponential Smoothing':
            er_forecast_ex, var_forecast_ex = es_forecast(excess_returns)
            optimal_weight = (1 / gamma) * (er_forecast_ex / var_forecast_ex)
            monthly_optimal_weights.append(optimal_weight)

            # Assuming reinvestment in the index with the calculated optimal weight
            next_month_return = data['Returns'][end_date] * optimal_weight
            monthly_returns.append(next_month_return)
            # Performance Metrics
            cumulative_return = np.prod(np.array(monthly_returns) + 1) - 1
            mean_return = np.mean(monthly_returns)
            std_return = np.std(monthly_returns)
            sharpe_ratio = mean_return / std_return * np.sqrt(12)  # Annualized Sharpe Ratio
        else:  # ML Forecast
            er_forecast_ml, var_forecast_ml, best_model_name = ml_forecast(excess_returns)
            print(f"Best Model: {best_model_name}, Forecast ER: {er_forecast_ml}, Forecast Variance: {var_forecast_ml}")
            optimal_weight = (1 / gamma) * (er_forecast_ml / var_forecast_ml)
            monthly_optimal_weights.append(optimal_weight)

            # Assuming reinvestment in the index with the calculated optimal weight
            next_month_return = data['Returns'][end_date] * optimal_weight
            monthly_returns.append(next_month_return)
            # Performance Metrics
            cumulative_return = np.prod(np.array(monthly_returns) + 1) - 1
            mean_return = np.mean(monthly_returns)
            std_return = np.std(monthly_returns)
            sharpe_ratio = mean_return / std_return * np.sqrt(12)  # Annualized Sharpe Ratio

    # Performance Metrics
    #cumulative_return = np.prod(np.array(monthly_returns) + 1) - 1
    #mean_return = np.mean(monthly_returns)
    #std_return = np.std(monthly_returns)
    #sharpe_ratio = mean_return / std_return * np.sqrt(12)  # Annualized Sharpe Ratio

    return monthly_optimal_weights, cumulative_return, sharpe_ratio


# Streamlit UI
def main():
    st.title("Portfolio Optimization Tool")

    # User inputs
    start_date, end_date = st.date_input("Select the date range for the data:",
                                         value=[pd.to_datetime('2013-01-01'), pd.to_datetime('2023-01-01')],
                                         min_value=pd.to_datetime('2000-01-01'), max_value=pd.to_datetime('2023-01-01'))
    in_sample_years = st.slider('In-sample period (years):', 1, 10, 5)
    gamma = st.slider('Risk Aversion Coefficient (γ):', 1.0, 10.0, 5.0)
    forecast_method = st.selectbox('Forecasting Method:', ['Classical:ARIMA', 'Classical:Exponential Smoothing', 'Machine Learning'])

    if st.button('Calculate Optimal Weights and Evaluate'):
        # Download and prepare data
        spx_data = download_data('^GSPC', start_date, end_date)
        gs1m_data = download_data('^IRX', start_date, end_date)
        spx_data['Monthly Yield'] = gs1m_data['Adj Close'] / 1200  # Convert annualized yield to monthly yield
        data = spx_data[['Returns', 'Monthly Yield']].dropna()

        # Calculate and evaluate
        monthly_optimal_weights, cumulative_return, sharpe_ratio = calculate_and_evaluate(data, gamma, in_sample_years,
                                                                                          forecast_method)

        # Display results

        st.line_chart(monthly_optimal_weights)
        st.write(f"Average Optimal Weight: {np.mean(monthly_optimal_weights):.2f}")
        st.write(f"Cumulative Return: {cumulative_return:.2f}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        #st.write(f"Best model for the machine learning was achieved by: {best_model_name}")



if __name__ == "__main__":
    main()
