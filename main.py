import warnings
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from strategy_engine import StrategyEngine

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    tickers = ["SPY", "^DJI", "^DJT"]
    loader = DataLoader(start_date="2020-01-01", split_date="2023-01-01", tickers=tickers)
    train, eval = loader.load_data()   

    fe = FeatureEngineer()
    train = fe.add_features(train, sma_n=20, span_n=10)
    eval = fe.add_features(eval, sma_n=20, span_n=10)

    strategy = StrategyEngine(threshold=2.0, leverage=2.0)
    train = strategy.create_signals(train)      
    train_bt = strategy.execute_backtest(train)  
    
    # Plot the equity curve
    plt.figure(figsize=(12,6))
    plt.plot(train_bt.index, train_bt["equity_curve"], label="Equity Curve")
    plt.title("Backtest Equity Curve")
    plt.legend()
    #plt.show()
