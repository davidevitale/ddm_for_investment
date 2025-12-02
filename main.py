import pandas as pd
from typing import Dict
import logging

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from strategy_engine import StrategyEngine
from genetic_optimizer import GeneticOptimizer
from performance_analyzer import PerformanceAnalyzer
from plot import (
    plot_equity_curves,
    plot_drawdowns,
    plot_return_vs_volatility,
    plot_rolling_sharpe
)

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

tickers_list = [
    ["SPY", "^DJI", "^DJT"],  # Pipeline for SPY
    ["QQQ", "^DJI", "^DJT"],  # Pipeline for QQQ
]
start_date = "2007-04-01"
split_date = "2022-01-01"

results = {}
metrics_all = {}

def evaluate_and_report(data: pd.DataFrame, params: Dict, plot_title: str, main_ticker: str):
    """
    Evaluate a trading strategy on given dataset, generate signals, run backtest, calculate performance metrics, plot equity curve and store results.
    """
    fe = FeatureEngineer()
    df_proc = fe.add_features(data, params['sma_n'], params['span_n'], main_ticker=main_ticker)

    se = StrategyEngine(threshold=2.0, main_ticker=main_ticker)
    df_sig = se.create_signals(df_proc)
    res = se.execute_backtest(df_sig)

    metrics_filename = f"{main_ticker.lower()}_metrics.csv"
    metrics = PerformanceAnalyzer.calculate_metrics(res, save_csv=True, filename=metrics_filename)

    logging.info("Performance Metrics for %s:", main_ticker)
    for k, v in metrics.items():
        logging.info("%s: %s", k, v)

    PerformanceAnalyzer.plot_equity(res, plot_title, main_ticker=main_ticker)

    results[main_ticker] = res
    metrics_all[main_ticker] = metrics

def run_pipeline(tickers: list, main_ticker: str):
    """
    Run the full trading strategy pipeline for a given main ticker.
    """
    logging.info("Running pipeline for %s", main_ticker)

    loader = DataLoader(start_date, split_date, tickers)
    df_train, df_eval = loader.get_data(
        columns=[f"{main_ticker}_Close", f"{main_ticker}_Volume"],
        main_ticker=main_ticker
    )

    optimizer = GeneticOptimizer(df_train, main_ticker=main_ticker,
                                 population_size=40, mutation_rate=0.2, crossover_rate=0.9)
    best_params = optimizer.run(generations=40)

    logging.info("--- TRAINING SET EVALUATION for %s ---", main_ticker)
    evaluate_and_report(df_train, best_params, f"{main_ticker} Training Equity Curve", main_ticker)

    logging.info("--- TEST SET EVALUATION for %s ---", main_ticker)
    evaluate_and_report(df_eval, best_params, f"{main_ticker} Test Equity Curve", main_ticker)

if __name__ == "__main__":
    for tickers in tickers_list:
        main_ticker = tickers[0]
        run_pipeline(tickers, main_ticker)

    plot_equity_curves(results["SPY"], results["QQQ"])
    plot_drawdowns(results["SPY"], results["QQQ"])
    plot_return_vs_volatility(metrics_all["SPY"], metrics_all["QQQ"])
    plot_rolling_sharpe(results["SPY"], results["QQQ"])
