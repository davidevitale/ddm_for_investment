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
        start_price = spy_df["SPY_Close"].iloc[0]
        benchmark = (spy_df["SPY_Close"] / start_price) * 100
        axes[0].plot(spy_df.index, benchmark, label="SPY Buy&Hold", color="gray", linestyle="--")
    axes[0].set_title("SPY Equity Curve")
    axes[0].legend()

    # Plot QQQ strategy equity curve
    axes[1].plot(qqq_df.index, qqq_df['equity_curve'], label="QQQ Strategy", color="black", linewidth=2)
    if "QQQ_Close" in qqq_df.columns:
        start_price = qqq_df["QQQ_Close"].iloc[0]
        benchmark = (qqq_df["QQQ_Close"] / start_price) * 100
        axes[1].plot(qqq_df.index, benchmark, label="QQQ Buy&Hold", color="gray", linestyle="--")
    axes[1].set_title("QQQ Equity Curve")
    axes[1].legend()

    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = "equity_curves_comparison.png"
    filepath = os.path.join(results_dir, filename)

    plt.savefig(filepath, dpi=300)
    print(f"Plot salvato in: {filepath}")