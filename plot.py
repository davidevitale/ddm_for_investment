import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_equity_curves(spy_df: pd.DataFrame, qqq_df: pd.DataFrame):
    """
    Compare Strategy Equity vs Benchmark for SPY and QQQ.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot SPY strategy equity curve
    axes[0].plot(spy_df.index, spy_df['equity_curve'], label="SPY Strategy", color="black", linewidth=2)
    if "SPY_Close" in spy_df.columns:
        # Calculate SPY Buy&Hold benchmark rebased to 100
        start_price = spy_df["SPY_Close"].iloc[0]
        benchmark = (spy_df["SPY_Close"] / start_price) * 100
        axes[0].plot(spy_df.index, benchmark, label="SPY Buy&Hold", color="gray", linestyle="--")
    axes[0].set_title("SPY Equity Curve")
    axes[0].legend()

    # Plot QQQ strategy equity curve
    axes[1].plot(qqq_df.index, qqq_df['equity_curve'], label="QQQ Strategy", color="black", linewidth=2)
    if "QQQ_Close" in qqq_df.columns:
        # Calculate QQQ Buy&Hold benchmark rebased to 100
        start_price = qqq_df["QQQ_Close"].iloc[0]
        benchmark = (qqq_df["QQQ_Close"] / start_price) * 100
        axes[1].plot(qqq_df.index, benchmark, label="QQQ Buy&Hold", color="gray", linestyle="--")
    axes[1].set_title("QQQ Equity Curve")
    axes[1].legend()

    plt.tight_layout()
    filename = "equity_curves_comparison.png"
    plt.savefig(os.path.join(os.getcwd(), filename), dpi=300)

def plot_drawdowns(spy_df: pd.DataFrame, qqq_df: pd.DataFrame):
    """
    Compare drawdown percentages between SPY and QQQ.
    """
    plt.figure(figsize=(14, 6))
    # SPY drawdown line
    plt.plot(spy_df.index, spy_df["drawdown"], label="SPY Drawdown", color="blue")
    # QQQ drawdown line
    plt.plot(qqq_df.index, qqq_df["drawdown"], label="QQQ Drawdown", color="red")
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown %")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    filename = "drawdown_comparison.png"
    plt.savefig(os.path.join(os.getcwd(), filename), dpi=300)

def plot_return_vs_volatility(spy_metrics: dict, qqq_metrics: dict):
    """
    Scatter plot of annualized return vs annualized volatility.
    """
    plt.figure(figsize=(8, 6))
    # SPY point
    plt.scatter(spy_metrics["annualized_volatility %"], spy_metrics["annualized_return %"],
                color="blue", label="SPY", s=100)
    # QQQ point
    plt.scatter(qqq_metrics["annualized_volatility %"], qqq_metrics["annualized_return %"],
                color="red", label="QQQ", s=100)

    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    plt.title("Return vs Volatility")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    filename = "return_vs_volatility.png"
    plt.savefig(os.path.join(os.getcwd(), filename), dpi=300)

def plot_rolling_sharpe(spy_df: pd.DataFrame, qqq_df: pd.DataFrame, window=126):
    """
    Rolling Sharpe ratio (e.g. 6 months ~ 126 trading days).
    """
    # Calculate daily returns for SPY and QQQ strategy equity curves
    spy_returns = spy_df["equity_curve"].pct_change().dropna()
    qqq_returns = qqq_df["equity_curve"].pct_change().dropna()

    # Rolling Sharpe ratio calculation (mean/std * sqrt(252))
    spy_rolling = (spy_returns.rolling(window).mean() / spy_returns.rolling(window).std()) * (252**0.5)
    qqq_rolling = (qqq_returns.rolling(window).mean() / qqq_returns.rolling(window).std()) * (252**0.5)

    plt.figure(figsize=(14, 6))
    plt.plot(spy_rolling.index, spy_rolling, label="SPY Rolling Sharpe", color="blue")
    plt.plot(qqq_rolling.index, qqq_rolling, label="QQQ Rolling Sharpe", color="red")
    plt.title(f"Rolling Sharpe Ratio (window={window} days)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    filename = "rolling_sharpe.png"
    plt.savefig(os.path.join(os.getcwd(), filename), dpi=300)
