import pandas as pd
from prophet import Prophet
import yfinance as yf
import pickle

def fetch_data(ticker='ETH-USD', start_date=None, end_date=None):
    eth_data = yf.download(ticker, start=start_date, end=end_date)
    eth_data.reset_index(inplace=True)
    df = eth_data[['Date', 'Close']]
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    return df

def check_stationarity(df):
    result = adfuller(df['y'])
    return result[0], result[1]

def train_model(df):
    model = Prophet()
    # model.add_country_holidays(country_name='US')
    model.fit(df)
    return model

def save_model(model, filename='model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename='model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, future_periods=180):
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    return forecast

def cross_validate_model(model, initial='90 days', period='180 days', horizon='180 days'):
    from prophet.diagnostics import cross_validation, performance_metrics
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)
    return df_p
