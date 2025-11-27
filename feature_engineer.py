import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame, sma_n: int, span_n: int) -> pd.DataFrame:
        """
        Add technical features to the DataFrame.
        """
        out_df = df.copy()
        
        # Simple Moving Average of 'Difference'
        out_df['SMA'] = out_df['Difference'].rolling(window=sma_n).mean()
        
        # Standard Deviation of 'Difference'
        out_df["STDEV"] = out_df["Difference"].rolling(sma_n).std(ddof=1).replace(0, np.nan)
        
        # Z-Score calculation
        out_df['z_score'] = (out_df['Difference'] - out_df['SMA']) / out_df['STDEV']
        
        # Exponential Moving Average of SPY Volume
        out_df['EMA'] = out_df['SPY_Volume'].ewm(span=span_n, adjust=False).mean()
        
        return out_df
