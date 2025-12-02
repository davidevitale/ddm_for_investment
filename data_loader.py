from typing import List, Tuple
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataLoader:
    """
    DataLoader class for downloading and preparing financial time series data.
    """
    def __init__(self, start_date: str, split_date: str, tickers: List[str]):
        self.start_date = start_date
        self.split_date = split_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.tickers = tickers
        
    def get_data(self, columns: List[str], main_ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download and prepare financial data for the given tickers.
        """
        try:
            data_dict = {}
            for ticker in self.tickers:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    progress=False
                )
                # Clean multi-level header if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                data_dict[ticker] = df
        except Exception as e:
            raise ConnectionError(f"Error downloading data: {e}")

        merged = pd.DataFrame(index=data_dict[self.tickers[0]].index)

        for ticker, df in data_dict.items():
            if ticker == main_ticker:
                merged[f"{ticker}_Close"] = df["Close"]
                merged[f"{ticker}_Volume"] = df["Volume"]
            else:
                merged[f"{ticker}_Close"] = df["Close"]

        # Compute Difference between DJI and DJT
        if "^DJI" in data_dict and "^DJT" in data_dict:
            merged["Difference"] = merged["^DJI_Close"] - merged["^DJT_Close"]

        # Keep only selected columns (excluding DJI_Close and DJT_Close)
        merged = merged[columns + ["Difference"]]

        # Drop initial NaN
        merged.dropna(inplace=True)

        # Split Train / Eval
        df_train = merged[merged.index < self.split_date].copy()
        df_eval = merged[merged.index >= self.split_date].copy()

        print(f"Data downloaded. Train set: {len(df_train)} rows, Eval set: {len(df_eval)} rows.")
        return df_train, df_eval
