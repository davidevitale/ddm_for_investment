import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import random
import warnings

# Ignora i warning per pulizia output
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADER
# ==========================================
class DataLoader:
    def __init__(self, start_date: str, split_date: str, tickers: List[str]):
        self.start_date = start_date
        self.split_date = split_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.tickers = tickers
        
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Scaricamento dati in corso...")
        # Tickers: SPY, ^DJI, ^DJT
        t1, t2, t3 = self.tickers
        
        try:
            data1 = yf.download(t1, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)[['Close', 'Volume']]
            data2 = yf.download(t2, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)[['Close']]
            data3 = yf.download(t3, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)[['Close']]
        except Exception as e:
            raise ConnectionError(f"Errore nel download dei dati: {e}")

        # Pulizia header multi-livello se presente
        for d in [data1, data2, data3]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)

        # Unione DataFrame
        data = pd.concat([data1['Close'], data1['Volume'], data2['Close'], data3['Close']], axis=1)
        data.columns = ['SPY_Close', 'SPY_Volume', 'DJI_Close', 'DJT_Close']
        
        # Calcolo Differenza
        data['Difference'] = data['DJI_Close'] - data['DJT_Close']
        data = data.drop(columns=['DJI_Close', 'DJT_Close'])
        
        # Drop NaN iniziali
        data.dropna(inplace=True)
        
        # Split Train / Eval
        df_train = data[data.index < self.split_date].copy()
        df_eval = data[data.index >= self.split_date].copy()
        
        print(f"Dati scaricati. Train set: {len(df_train)} righe, Eval set: {len(df_eval)} righe.")
        return df_train, df_eval

# ==========================================
# 2. FEATURE ENGINEER
# ==========================================
class FeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame, sma_n: int, span_n: int) -> pd.DataFrame:
        out_df = df.copy()
        
        # Rolling Mean e StdDev sulla 'Difference'
        out_df['SMA'] = out_df['Difference'].rolling(window=sma_n).mean()
        out_df["STDEV"] = out_df["Difference"].rolling(sma_n).std(ddof=1).replace(0, np.nan)
        
        # Z-Score
        out_df['z_score'] = (out_df['Difference'] - out_df['SMA']) / out_df['STDEV']
        
        # EMA sui Volumi
        out_df['EMA'] = out_df['SPY_Volume'].ewm(span=span_n, adjust=False).mean()
        
        return out_df

# ==========================================
# 3. STRATEGY ENGINE
# ==========================================
class StrategyEngine:
    def __init__(self, threshold: float = 2.0, leverage: float = 2.0):
        self.threshold = threshold
        self.leverage = leverage

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        sig = pd.Series(0, index=out_df.index, dtype="int8")
        
        # Logica di ingresso
        cond_buy  = (out_df["z_score"] < -self.threshold) & (out_df["SPY_Volume"] > out_df["EMA"])
        cond_sell = (out_df["z_score"] >  self.threshold) & (out_df["SPY_Volume"] < out_df["EMA"])
        
        sig[cond_buy]  =  1
        sig[cond_sell] = -1
        
        # Gestione NaN
        sig[out_df["z_score"].isna()] = 0
        
        out_df['Signal'] = sig
        return out_df

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["Position"] = result["Signal"].shift(1).fillna(0)
        
        result["Entry_Price"] = 0.0
        result["Profit"] = 0.0
        result["MToM"] = 0.0
        
        # Array numpy per iterazione
        idx = result.index
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
            
            if pos == 1: # LONG
                if prev_pos == 0: 
                    curr_entry = prev_px
                elif prev_pos == -1: # Reverse
                    curr_profit = (entry_price[i-1] - prev_px) / entry_price[i-1] * 100 * self.leverage
                    curr_entry = prev_px
                elif prev_pos == 1: # Hold
                    curr_entry = entry_price[i-1]
                
                if curr_entry != 0:
                    curr_mtom = (px - curr_entry) / curr_entry * 100 * self.leverage

            elif pos == -1: # SHORT
                if prev_pos == 0:
                    curr_entry = prev_px
                elif prev_pos == 1: # Reverse
                    curr_profit = (prev_px - entry_price[i-1]) / entry_price[i-1] * 100 * self.leverage
                    curr_entry = prev_px
                elif prev_pos == -1: # Hold
                    curr_entry = entry_price[i-1]
                
                if curr_entry != 0:
                    curr_mtom = (curr_entry - px) / curr_entry * 100 * self.leverage

            else: # FLAT
                if prev_pos == -1: # Close Short
                    curr_profit = (entry_price[i-1] - prev_px) / entry_price[i-1] * 100 * self.leverage
                elif prev_pos == 1: # Close Long
                    curr_profit = (prev_px - entry_price[i-1]) / entry_price[i-1] * 100 * self.leverage
            
            entry_price[i] = curr_entry
            profit[i] = curr_profit
            mtom[i] = curr_mtom

        result["Entry_Price"] = entry_price
        result["Profit"] = profit
        result["MToM"] = mtom
        
        result["Cumulative_Profit"] = result["Profit"].cumsum()
        result["equity_curve"] = result["Cumulative_Profit"] + result["MToM"] + 100
        
        result["max_equity"] = result["equity_curve"].cummax()
        result["drawdown"] = (result["max_equity"] - result["equity_curve"]) / result["max_equity"] * 100
        
        dd_series = result["drawdown"]
        result["drawdown_length"] = (dd_series > 0).astype(int).groupby((dd_series == 0).cumsum()).cumsum()
        
        return result

# ==========================================
# 4. GENETIC OPTIMIZER
# ==========================================
class GeneticOptimizer:
    def __init__(self, data: pd.DataFrame, population_size=50, mutation_rate=0.2, crossover_rate=0.85):
        self.data = data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.parameter_bounds = {
            'sma_n': [5, 50],
            'span_n': [5, 50],
        }
        self.parameter_names = list(self.parameter_bounds.keys())
        self.feature_engine = FeatureEngineer()
        self.strategy_engine = StrategyEngine()

    def _create_individual(self) -> Dict[str, int]:
        return {
            k: random.randint(v[0], v[1]) for k, v in self.parameter_bounds.items()
        }

    def _fitness_function(self, individual: Dict[str, int]) -> float:
        try:
            df_temp = self.feature_engine.add_features(self.data, individual['sma_n'], individual['span_n'])
            df_temp = self.strategy_engine.generate_signals(df_temp)
            res = self.strategy_engine.run_backtest(df_temp)
            
            final_equity = res['equity_curve'].iloc[-1]
            
            drawdowns = res["drawdown"]
            non_zero_drawdowns = drawdowns[drawdowns > 0]
            mean_drawdown = non_zero_drawdowns.mean() if len(non_zero_drawdowns) > 0 else 0
            
            return final_equity - mean_drawdown
        except:
            return -1000.0

    def _crossover(self, p1: Dict, p2: Dict) -> Tuple[Dict, Dict]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        
        off1, off2 = p1.copy(), p2.copy()
        off1['span_n'] = p2['span_n']
        off2['span_n'] = p1['span_n']
        return off1, off2

    def _mutate(self, ind: Dict) -> Dict:
        mutated = ind.copy()
        for param in self.parameter_names:
            if random.random() < self.mutation_rate:
                bounds = self.parameter_bounds[param]
                change = random.randint(-3, 3)
                new_val = mutated[param] + change
                mutated[param] = max(bounds[0], min(bounds[1], new_val))
        return mutated

    def run(self, generations=10) -> Dict[str, Any]:
        print(f"Inizio ottimizzazione genetica ({generations} generazioni)...")
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_global_ind = None
        best_global_fit = float('-inf')
        
        for gen in range(generations):
            fitnesses = [self._fitness_function(ind) for ind in population]
            
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            
            if max_fit > best_global_fit:
                best_global_fit = max_fit
                best_global_ind = population[best_idx].copy()
            
            print(f"Gen {gen+1}/{generations} | Best Fit: {max_fit:.2f} | Best Params: {population[best_idx]}")
            
            sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
            
            new_pop = []
            new_pop.append(sorted_pop[0]) # Elitismo
            new_pop.append(sorted_pop[1])
            
            while len(new_pop) < self.population_size:
                p1 = random.choice(sorted_pop[:int(self.population_size/2)])
                p2 = random.choice(sorted_pop[:int(self.population_size/2)])
                
                o1, o2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(o1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(o2))
            
            population = new_pop

        print(f"\nOttimizzazione completata. Migliori parametri: {best_global_ind}")
        return best_global_ind

# ==========================================
# 5. PERFORMANCE ANALYZER (CORRETTO E RIGOROSO)
# ==========================================
class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(strategy_df: pd.DataFrame, initial_capital=100.0) -> Dict[str, float]:
        final_equity = strategy_df["equity_curve"].iloc[-1]
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
        max_dd = strategy_df["drawdown"].max()
        
        # --- 1. CALCOLO CAGR (Rendimento Annualizzato) ---
        if not isinstance(strategy_df.index, pd.DatetimeIndex):
             strategy_df.index = pd.to_datetime(strategy_df.index)
             
        start_date = strategy_df.index[0]
        end_date = strategy_df.index[-1]
        duration_days = (end_date - start_date).days
        duration_years = duration_days / 365.25
        
        cagr_pct = 0.0
        if duration_years > 0 and final_equity > 0:
            # Formula CAGR: (Finale / Iniziale)^(1/Anni) - 1
            cagr_val = (final_equity / initial_capital) ** (1 / duration_years) - 1
            cagr_pct = cagr_val * 100

        # --- 2. SHARPE RATIO (Corretto, basato su equity giornaliera) ---
        equity_daily_returns = strategy_df['equity_curve'].pct_change().dropna()
        
        if len(equity_daily_returns) > 1:
            # Volatilità annualizzata dei rendimenti giornalieri
            annual_std = equity_daily_returns.std() * np.sqrt(252)
            # Rendimento medio annualizzato
            annual_mean_ret = equity_daily_returns.mean() * 252
            
            risk_free = 0.015
            
            sharpe = (annual_mean_ret - risk_free) / annual_std if annual_std != 0 else 0
        else:
            sharpe = 0.0
        
        # --- 3. CALMAR RATIO (Corretto, basato su CAGR) ---
        if max_dd > 0:
            calmar = cagr_pct / max_dd
        else:
            # Se drawdown è 0 (impossibile ma teorico), diamo un valore alto o 0
            calmar = 0.0 

        # --- Statistiche Trade (conteggio e profit factor) ---
        trade_returns = strategy_df['Profit'][strategy_df['Profit'] != 0]
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = trade_returns[trade_returns < 0].abs().sum()
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        return {
            "Final Equity": round(final_equity, 2),
            "Total Return %": round(total_return_pct, 2),
            "CAGR %": round(cagr_pct, 2),      # Aggiunto per chiarezza
            "Max Drawdown %": round(max_dd, 2),
            "Calmar Ratio": round(calmar, 2),  # Ora è realistico (CAGR / MaxDD)
            "Sharpe Ratio": round(sharpe, 2),  # Ora è realistico (Daily Equity Returns)
            "Total Trades": len(trade_returns),
            "Profit Factor": round(profit_factor, 2),
            "annualized_volatility %": round(annual_std*100, 4),
            "annualized_return %": round(annual_mean_ret*100, 4)
        }

    @staticmethod
    def plot_equity(strategy_df: pd.DataFrame, title: str):
        plt.figure(figsize=(12, 6))
        
        # --- 1. Plot della Strategia ---
        plt.plot(strategy_df.index, strategy_df['equity_curve'], 
                 label='Strategy Equity', color='blue', linewidth=1.5)
        
        # --- 2. Plot del Benchmark (SPY) ---
        # Verifichiamo che la colonna SPY esista (dovrebbe esserci dal DataLoader)
        if 'SPY_Close' in strategy_df.columns:
            # Calcoliamo il fattore di normalizzazione per far partire l'SPY da 100
            # (100 è il capitale base usato nella tua StrategyEngine)
            initial_capital = 100.0
            start_price = strategy_df['SPY_Close'].iloc[0]
            
            # Formula: (Prezzo Corrente / Prezzo Iniziale) * Capitale Iniziale
            spy_equity = (strategy_df['SPY_Close'] / start_price) * initial_capital
            
            plt.plot(strategy_df.index, spy_equity, 
                     label='SPY (Buy & Hold)', color='gray', linestyle='--', alpha=0.7)

        # --- 3. Formattazione Grafico ---
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Equity (Rebased to 100)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best') # Posiziona la legenda nel punto migliore
        
        # Opzionale: Aggiungi riempimento tra le linee per evidenziare outperformance
        if 'SPY_Close' in strategy_df.columns:
             plt.fill_between(strategy_df.index, strategy_df['equity_curve'], spy_equity, 
                              where=(strategy_df['equity_curve'] > spy_equity),
                              color='green', alpha=0.1, interpolate=True)
             plt.fill_between(strategy_df.index, strategy_df['equity_curve'], spy_equity, 
                              where=(strategy_df['equity_curve'] <= spy_equity),
                              color='red', alpha=0.1, interpolate=True)

        plt.show()

# ==========================================
# 6. MAIN APPLICATION
# ==========================================
class MeanReversionApp:
    def __init__(self):
        self.tickers = ["SPY", "^DJI", "^DJT"]
        self.start_date = "2007-04-01"
        self.split_date = "2022-01-01"
        
    def run(self):
        # 1. Caricamento Dati
        loader = DataLoader(self.start_date, self.split_date, self.tickers)
        df_train, df_eval = loader.get_data()
        
        # 2. Ottimizzazione Genetica (Training)
        # FIX: generations spostato nel metodo run()
        optimizer = GeneticOptimizer(df_train, population_size=40, mutation_rate=0.2, crossover_rate=0.9) 
        best_params = optimizer.run(generations=40)
        
        # 3. Valutazione In-Sample (Training)
        print("\n--- VALUTAZIONE TRAINING SET ---")
        self._evaluate_and_report(df_train, best_params, "Training Equity Curve")
        
        # 4. Valutazione Out-of-Sample (Test)
        print("\n--- VALUTAZIONE TEST SET (Out-of-Sample) ---")
        self._evaluate_and_report(df_eval, best_params, "Test Equity Curve")

    def _evaluate_and_report(self, data: pd.DataFrame, params: Dict, plot_title: str):
        fe = FeatureEngineer()
        df_proc = fe.add_features(data, params['sma_n'], params['span_n'])
        
        se = StrategyEngine(threshold=2.0)
        df_sig = se.generate_signals(df_proc)
        res = se.run_backtest(df_sig)
        
        metrics = PerformanceAnalyzer.calculate_metrics(res)
        print("Performance Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        PerformanceAnalyzer.plot_equity(res, plot_title)

if __name__ == "__main__":
    app = MeanReversionApp()
    app.run()