import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    @staticmethod
    def add_features(
        df: pd.DataFrame,
        sma_n: int,
        span_n: int,
        main_ticker: str,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Add technical features to the DataFrame and save to /data/processed/.
        """
        out_df = df.copy()

        # Difference tra DJI e DJT
        if "^DJI_Close" in df.columns and "^DJT_Close" in df.columns:
            out_df["Difference"] = out_df["^DJI_Close"] - out_df["^DJT_Close"]

        # Simple Moving Average of 'Difference'
        out_df["SMA"] = out_df["Difference"].rolling(window=sma_n).mean()

        # Standard Deviation of 'Difference'
        out_df["STDEV"] = (
            out_df["Difference"].rolling(sma_n).std(ddof=1).replace(0, np.nan)
        )

        # Z-Score
        out_df["z_score"] = (out_df["Difference"] - out_df["SMA"]) / out_df["STDEV"]

        # Exponential Moving Average of main ticker Volume
        out_df["EMA"] = (
            out_df[f"{main_ticker}_Volume"].ewm(span=span_n, adjust=False).mean()
        )

        if save_path is None:
            save_path = os.path.join("data", "processed", f"{main_ticker}_preprocessed_data.csv")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_df.to_csv(save_path, index=True)

        return out_df