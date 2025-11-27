import warnings
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from strategy_engine import StrategyEngine
from genetic_optimizer import GeneticOptimizer
from performance_analyzer import PerformanceAnalyzer

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    tickers = ["SPY", "^DJI", "^DJT"]
    loader = DataLoader(start_date="2020-01-01", split_date="2023-01-01", tickers=tickers)
    train, eval = loader.load_data()

    # Genetic Optimization
    print("Starting genetic optimization (40 generations)...")
    optimizer = GeneticOptimizer(train, population_size=40, mutation_rate=0.2, crossover_rate=0.9)
    best_params = optimizer.run(generations=40)

    # Training Evaluation
    print("\n--- TRAINING SET EVALUATION ---")
    fe = FeatureEngineer()
    train_proc = fe.add_features(train, best_params['sma_n'], best_params['span_n'])
    se = StrategyEngine(threshold=2.0)
    train_sig = se.create_signals(train_proc)
    train_bt = se.execute_backtest(train_sig)
    metrics_train = PerformanceAnalyzer.calculate_metrics(train_bt)
    print("Performance Metrics:")
    for k, v in metrics_train.items():
        print(f"{k}: {v}")
    strategy = StrategyEngine(threshold=2.0, leverage=2.0)
    train = strategy.create_signals(train)      
    train_bt = strategy.execute_backtest(train) 
    
    # Test Evaluation
    print("\n--- TEST SET EVALUATION (Out-of-Sample) ---")
    eval_proc = fe.add_features(eval, best_params['sma_n'], best_params['span_n'])
    eval_sig = se.create_signals(eval_proc)
    eval_bt = se.execute_backtest(eval_sig)
    metrics_eval = PerformanceAnalyzer.calculate_metrics(eval_bt)
    print("Performance Metrics:")
    for k, v in metrics_eval.items():
        print(f"{k}: {v}")
        
    # Plot equity curve for training
    print("Plotting training equity curve...")
    plt.figure(figsize=(12,6))
    plt.plot(train_bt.index, train_bt["equity_curve"], label="Training Equity Curve")
    plt.title("Training Equity Curve")
    plt.legend()
    plt.show()

    # Plot equity curve for test
    print("Plotting test equity curve...")
    plt.figure(figsize=(12,6))
    plt.plot(eval_bt.index, eval_bt["equity_curve"], label="Test Equity Curve")
    plt.title("Test Equity Curve")
    plt.legend()
    # plt.show()
