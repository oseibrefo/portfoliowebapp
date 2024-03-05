#Portfolio Exercise by Emmanuel Osei-Brefo
#March 2024

import streamlit as st
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from fredapi import Fred
import pandas as pd



# Download historical data
def download_data(start_date, end_date ):
    # Replace 'YOUR_API_KEY' with your actual FRED API key
    api_key = '459ec9128dc2ea340eddfe499fea0ba2'
    fred = Fred(api_key=api_key)

    # Define the series IDs for GS1M (10-Year Treasury Constant Maturity Rate) and S&P 500 Index
    gs1m_series_id = 'GS1M'
    s500_series_id = 'SP500'

    # Retrieve data for GS1M with specified date range
    gs1m_data = fred.get_series(gs1m_series_id, start=start_date, end=end_date)

    # Retrieve data for S&P 500 with specified date range
    s500_data = fred.get_series(s500_series_id, start=start_date, end=end_date)

    # Create DataFrames for GS1M and S&P 500 data
    gs1m_df = pd.DataFrame(gs1m_data, columns=['GS1M'])
    s500_df = pd.DataFrame(s500_data, columns=['SPX'])


    # Merge the two dataframes on date index
    data = pd.merge(s500_df, gs1m_df,  how='inner', left_index=True, right_index=True)
    return(data)

#Load RAW Data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

#Load and clean data
def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data_cleaned = df.dropna(how='all')
        data_cleaned.reset_index(drop=True, inplace=True)

        # Drop columns where all cells are NaN
        #df = data_cleaned.dropna(axis='columns', how='all')
        data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')
        data_cleaned = data_cleaned.dropna(subset=['Date'])
        data = data_cleaned.drop(columns=['Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        return data
    else:
        return None






#classical Forecasting Methods
#Exponential Smoothing Method
def es_forecast(data):
    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    er_forecast_ex = model_fit.forecast()#[0]
    var_forecast_ex = np.var(data)  # Simplification
    return er_forecast_ex, var_forecast_ex


#Arima Method
def arima_forecast(data):
    model_returns = ARIMA(data, order=(1, 1, 1))
    model_fit_returns = model_returns.fit()
    er_forecast_ar = model_fit_returns.forecast()
    residuals_squared = model_fit_returns.resid**2
    model_variance = ARIMA(residuals_squared, order=(1, 1, 1))
    model_fit_variance = model_variance.fit()
    var_forecast_ar = model_fit_variance.forecast()
    return er_forecast_ar, var_forecast_ar


# Historical forecasting method
def historical_forecast(data):
    er_forecast = np.mean(data)
    var_forecast = np.var(data)
    return er_forecast, var_forecast


# Machine learning forecasting methods
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

    # The best model is used to forecast the next return value
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
    return monthly_optimal_weights, cumulative_return, sharpe_ratio


# Streamlit User Interface Development

def main():
    st.title("Portfolio Optimisation Tool")

    data_source = st.radio('Select data source', ('Upload CSV', 'FRED API'))

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            #Upload and clean data
            data = load_and_clean_data(uploaded_file)
            # Optionally display the uploaded data before cleaning
            if st.button('Clean and show csv data'):
                st.dataframe(data)




    else:
        st.subheader('Enter details to fetch the data using FRED API ')

        start_date, end_date = st.date_input("Select the date range for the data:",
                                         value=[pd.to_datetime('2014-01-01'), pd.to_datetime('2023-12-01')],
                                         min_value=pd.to_datetime('2000-01-01'), max_value=pd.to_datetime('2023-12-01'))
        data = download_data(start_date, end_date)
        #display data
        #st.write(data)
        if st.checkbox('Show raw API data'):
            st.write("Raw Data")
            st.dataframe(data)


    #if data is not None:
    in_sample_years = st.slider('In-sample period (years) :', 1, 10, 5)
    gamma = st.slider('Risk Aversion Coefficient (γ):', 1.0, 10.0, 5.0)
    forecast_method = st.selectbox('Forecasting Method:', ['Classical:ARIMA', 'Classical:Exponential Smoothing', 'Machine Learning'])

    if st.button('Calculate Optimal Weights and Evaluate'):
        data['Returns'] = data['SPX'].pct_change()  # Calculate SP500 monthly returns
        data['Monthly Yield'] = data['GS1M'] / 1200  # Convert annualized GS1M rates to monthly
        data = data[['Returns', 'Monthly Yield']].dropna()
        st.write(data)

            #Calculate and evaluate the portfolio performance metrics and monthly optimal weights
        monthly_optimal_weights, cumulative_return, sharpe_ratio = calculate_and_evaluate(data, gamma, in_sample_years,
                                                                                              forecast_method)

            # Display results

        st.line_chart(monthly_optimal_weights)
        st.write(f"Average Optimal Weight: {np.mean(monthly_optimal_weights):.2f}")
        st.write(f"Cumulative Return: {cumulative_return:.2f}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()


