import yfinance as yf
import pandas as pd
from datetime import datetime

class DataLoader:
    def __init__(self, start_date: str, split_date: str, tickers: list[str]):
        self.start_date = start_date
        self.split_date = split_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.tickers = tickers
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("Downloading data...")
        t1, t2, t3 = self.tickers
        
        try:
            data1 = yf.download(t1, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)[['Close', 'Volume']]
            data2 = yf.download(t2, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)[['Close']]
            data3 = yf.download(t3, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)[['Close']]
        except Exception as e:
            raise ConnectionError(f"Error downloading data: {e}")

        # Clean multi-index headers if present
        for d in [data1, data2, data3]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)

        # Merge DataFrames
        data = pd.concat([data1['Close'], data1['Volume'], data2['Close'], data3['Close']], axis=1)
        data.columns = ['SPY_Close', 'SPY_Volume', 'DJI_Close', 'DJT_Close']
        
        # Calculate Difference
        data['Difference'] = data['DJI_Close'] - data['DJT_Close']
        data = data.drop(columns=['DJI_Close', 'DJT_Close'])
        data.dropna(inplace=True)
        
        # Train / Eval split
        df_train = data[data.index < self.split_date].copy()
        df_eval = data[data.index >= self.split_date].copy()
        
        print(f"Data downloaded. Train set: {len(df_train)} rows, Eval set: {len(df_eval)} rows.")
        return df_train, df_eval
