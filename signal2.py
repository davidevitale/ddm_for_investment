import pandas as pd
import numpy as np

class Signal:

    def __init__(self):
        # L'init può rimanere vuoto se la classe è solo un contenitore di metodi
        pass
    
    @staticmethod
    def make_signal_from_diff(df, col_zs, col_vol, col_ema, threshold=2) -> pd.Series:
        """
        Builds discrete signals in {-1, 0, +1} using Zscore and Volumes threshold.
        """
        # --- CORREZIONE 1: Dtype "float" per accettare np.nan ---
        sig = pd.Series(0.0, index=df.index, dtype="float")
        
        # --- CORREZIONE 2: Usare '&' per E-bitwise e parentesi () ---
        # Condizione per segnale SELL (-1)
        sell_condition = (df[col_zs] > threshold) & (df[col_vol] < df[col_ema])
        
        # Condizione per segnale BUY (1)
        buy_condition = (df[col_zs] < -threshold) & (df[col_vol] > df[col_ema])
        
        sig[sell_condition] = -1.0
        sig[buy_condition] = 1.0
        
        # Gestione NaN (migliorata)
        nan_mask = df[col_zs].isna() | df[col_vol].isna() | df[col_ema].isna()
        sig[nan_mask] = np.nan
        
        return sig
    
    @staticmethod
    def implement_trading_strategy(df: pd.DataFrame) -> pd.DataFrame:
        """
        State-machine strategy:
        - La posizione è basata sul segnale del giorno precedente.
        - Calcola PnL realizzato ('Profit') e PnL non realizzato ('MToM').
        """
        # Questa funzione era logicamente corretta, l'errore era nel chiamarla.
        # Aggiunto solo @staticmethod.
        
        result = df.copy()
        
        # Assicurati che la colonna "Signal" esista
        if "Signal" not in result.columns:
            raise ValueError("DataFrame 'df' non contiene la colonna 'Signal'. Esegui prima make_signal_from_diff.")
            
        result["Position"] = result["Signal"].shift(1).fillna(0) # position is based on previous signal
        result["Entry_Price"] = 0.0
        result["Profit"] = 0.0
        result["MToM"] = 0.0

        for i in range(1, len(result)):
            px = result.iloc[i]["Close"]
            previous_px = result.iloc[i-1]["Close"]
            pos = result.iloc[i]["Position"]
            previous_pos = result.iloc[i-1]["Position"]

            if pos == 1:
                if previous_pos == 0:  # Entering long
                    result.at[i, "Entry_Price"] = previous_px
                if previous_pos == -1:  # Exiting short and entering long
                    result.at[i, "Profit"] = (result.at[i-1, "Entry_Price"] - previous_px) / result.at[i-1, "Entry_Price"] * 100
                    result.at[i, "Entry_Price"] = previous_px
                if previous_pos == 1:  # Continuing long
                    result.at[i, "Entry_Price"] = result.at[i-1, "Entry_Price"]
                result.at[i, "MToM"] = (px - result.at[i, "Entry_Price"]) / result.at[i, "Entry_Price"] * 100
            elif pos == -1:
                if previous_pos == 0:  # Entering short
                    result.at[i, "Entry_Price"] = previous_px
                if previous_pos == 1:  # Exiting long and entering short
                    result.at[i, "Profit"] = (previous_px - result.at[i-1, "Entry_Price"]) / result.at[i-1, "Entry_Price"] * 100
                    result.at[i, "Entry_Price"] = previous_px
                if previous_pos == -1:  # Continuing short
                    result.at[i, "Entry_Price"] = result.at[i-1, "Entry_Price"]
                result.at[i, "MToM"] = (result.at[i, "Entry_Price"] - px) / result.at[i, "Entry_Price"] * 100
            else:
                if previous_pos == -1:  # Exiting short
                    result.at[i, "Profit"] = (result.at[i-1, "Entry_Price"] - previous_px) / result.at[i-1, "Entry_Price"] * 100
                if previous_pos == 1:  # Exiting long
                    result.at[i, "Profit"] = (previous_px - result.at[i-1, "Entry_Price"]) / result.at[i-1, "Entry_Price"] * 100