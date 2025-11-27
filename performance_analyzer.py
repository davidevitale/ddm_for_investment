import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(strategy_df: pd.DataFrame, initial_capital=100.0) -> Dict[str, float]:
        final_equity = strategy_df["equity_curve"].iloc[-1]
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
        max_dd = strategy_df["drawdown"].max()
        
        if not isinstance(strategy_df.index, pd.DatetimeIndex):
             strategy_df.index = pd.to_datetime(strategy_df.index)
             
        start_date = strategy_df.index[0]
        end_date = strategy_df.index[-1]
        duration_days = (end_date - start_date).days
        duration_years = duration_days / 365.25
        
        cagr_pct = 0.0
        if duration_years > 0 and final_equity > 0:
            cagr_val = (final_equity / initial_capital) ** (1 / duration_years) - 1
            cagr_pct = cagr_val * 100

        equity_daily_returns = strategy_df['equity_curve'].pct_change().dropna()
        
        if len(equity_daily_returns) > 1:
            annual_std = equity_daily_returns.std() * np.sqrt(252)
            annual_mean_ret = equity_daily_returns.mean() * 252
            risk_free = 0.015
            sharpe = (annual_mean_ret - risk_free) / annual_std if annual_std != 0 else 0
        else:
            sharpe = 0.0
        
        if max_dd > 0:
            calmar = cagr_pct / max_dd
        else:
            calmar = 0.0 

        trade_returns = strategy_df['Profit'][strategy_df['Profit'] != 0]
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = trade_returns[trade_returns < 0].abs().sum()
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        return {
            "Final Equity": round(final_equity, 2),
            "Total Return %": round(total_return_pct, 2),
            "CAGR %": round(cagr_pct, 2),
            "Max Drawdown %": round(max_dd, 2),
            "Calmar Ratio": round(calmar, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Total Trades": len(trade_returns),
            "Profit Factor": round(profit_factor, 2),
            "annualized_volatility %": round(annual_std*100, 4),
            "annualized_return %": round(annual_mean_ret*100, 4)
        }

    @staticmethod
    def plot_equity(strategy_df: pd.DataFrame, title: str):
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_df.index, strategy_df['equity_curve'], label='Equity Curve', color='blue')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
