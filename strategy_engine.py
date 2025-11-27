import numpy as np
import pandas as pd

class StrategyEngine:
    """
    StrategyEngine class to generate trading signals and execute backtests.
    """
    def __init__(self, threshold: float = 2.0, leverage: float = 2.0):
        self.threshold = threshold
        self.leverage = leverage

    def create_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on z-score and volume conditions.
        """
        out_df = df.copy()
        sig = pd.Series(0, index=out_df.index, dtype="int8")
        
        cond_buy  = (out_df["z_score"] < -self.threshold) & (out_df["SPY_Volume"] > out_df["EMA"])
        cond_sell = (out_df["z_score"] >  self.threshold) & (out_df["SPY_Volume"] < out_df["EMA"])
        
        sig[cond_buy]  =  1
        sig[cond_sell] = -1
        sig[out_df["z_score"].isna()] = 0
        
        out_df['Signal'] = sig
        return out_df

    def execute_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute backtest based on generated signals and calculate performance metrics.
        """
        result = df.copy()
        result["Position"] = result["Signal"].shift(1).fillna(0)
        
        result["Entry_Price"] = 0.0
        result["Profit"] = 0.0
        result["MToM"] = 0.0
        
        spy_close = result["SPY_Close"].values
        position = result["Position"].values
        entry_price = np.zeros(len(result))
        profit = np.zeros(len(result))
        mtom = np.zeros(len(result))
        
        for i in range(1, len(result)):
            px = spy_close[i]
            prev_px = spy_close[i-1]
            pos = position[i]
            prev_pos = position[i-1]
            
            curr_entry = 0.0
            curr_profit = 0.0
            curr_mtom = 0.0
            
            if pos == 1:  # LONG
                if prev_pos == 0:  # New long entry
                    curr_entry = prev_px
                elif prev_pos == -1:  # Reverse from short to long
                    curr_profit = (entry_price[i-1] - prev_px) / entry_price[i-1] * 100 * self.leverage
                    curr_entry = prev_px
                elif prev_pos == 1:  # Hold long
                    curr_entry = entry_price[i-1]
                if curr_entry != 0:
                    curr_mtom = (px - curr_entry) / curr_entry * 100 * self.leverage

            elif pos == -1:  # SHORT
                if prev_pos == 0:  # New short entry
                    curr_entry = prev_px
                elif prev_pos == 1:  # Reverse from long to short
                    curr_profit = (prev_px - entry_price[i-1]) / entry_price[i-1] * 100 * self.leverage
                    curr_entry = prev_px
                elif prev_pos == -1:  # Hold short
                    curr_entry = entry_price[i-1]
                if curr_entry != 0:
                    curr_mtom = (curr_entry - px) / curr_entry * 100 * self.leverage

            else:  # FLAT (no position)
                if prev_pos == -1:  # Closing short
                    curr_profit = (entry_price[i-1] - prev_px) / entry_price[i-1] * 100 * self.leverage
                elif prev_pos == 1:  # Closing long
                    curr_profit = (prev_px - entry_price[i-1]) / entry_price[i-1] * 100 * self.leverage
            
            entry_price[i] = curr_entry
            profit[i] = curr_profit
            mtom[i] = curr_mtom

        result["Entry_Price"] = entry_price
        result["Profit"] = profit
        result["MToM"] = mtom
        
        # Equity curve and risk metrics
        result["Cumulative_Profit"] = result["Profit"].cumsum()
        result["equity_curve"] = result["Cumulative_Profit"] + result["MToM"] + 100
        result["max_equity"] = result["equity_curve"].cummax()
        result["drawdown"] = (result["max_equity"] - result["equity_curve"]) / result["max_equity"] * 100
        
        dd_series = result["drawdown"]
        result["drawdown_length"] = (dd_series > 0).astype(int).groupby((dd_series == 0).cumsum()).cumsum()
        
        return result
