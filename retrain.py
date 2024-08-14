import pandas as pd
from model import fetch_data, train_model, save_model
from datetime import datetime

def main():
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=1)
    
    df = fetch_data(ticker='ETH-USD', start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    
    model = train_model(df)
    
    save_model(model) 
    
    print(f"Model retrained and saved successfully at {datetime.now()}")

if __name__ == '__main__':
    main()
