import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import random
import warnings

# Ignora avvisi comuni
warnings.filterwarnings('ignore')

# =========================
# 1. Caricamento e Preparazione Dati
# =========================

def load_and_prepare_data(start_date: str, 
                          end_date: str, 
                          split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scarica i dati da yfinance, calcola lo spread e divide in train/test.
    """
    print("Download dati in corso...")
    # Tickers
    ticker1 = "SPY"
    ticker2 = "^DJI" # Dow Jones Industrial
    ticker3 = "^DJT" # Dow Jones Transport

    # Scarica i dati
    data1 = yf.download(ticker1, start=start_date, end=end_date, auto_adjust=True)[['Close', 'Volume']]
    data2 = yf.download(ticker2, start=start_date, end=end_date, auto_adjust=True)[['Close']]
    data3 = yf.download(ticker3, start=start_date, end=end_date, auto_adjust=True)[['Close']]

    # Unisci tutto in un unico DataFrame
    data = pd.concat([data1['Close'], data1['Volume'], data2['Close'], data3['Close']], axis=1, join='inner')
    data.columns = ['SPY_Close', 'SPY_Volume', 'DJI_Close', 'DJT_Close']

    # Aggiungi la differenza (Spread)
    data['Spread'] = data['DJI_Close'] - data['DJT_Close']

    # Rinomina colonne per coerenza
    data = data.rename(columns={
        'SPY_Close': 'Adj Close',
        'SPY_Volume': 'Volume'
    })
    
    # Seleziona solo le colonne necessarie
    data = data[['Adj Close', 'Volume', 'Spread']]
    data = data.dropna()

    # Split in train / eval
    df_train = data[data.index < split_date].copy()
    df_test = data[data.index >= split_date].copy()
    
    print(f"Dati caricati: {len(data)} righe totali.")
    print(f"Training: {len(df_train)} righe ({df_train.index.min().date()} -> {df_train.index.max().date()})")
    print(f"Test: {len(df_test)} righe ({df_test.index.min().date()} -> {df_test.index.max().date()})")
    
    return df_train, df_test

# =========================
# 2. Costruzione del Segnale (Logica Z-Score aggiornata)
# =========================

class SignalBuilder:
    @staticmethod
    def make_signal(df: pd.DataFrame,
                    col_zs: str,       # Colonna Z-score (da Spread)
                    col_vol: str,      # Colonna Volume (da SPY_Volume)
                    col_ema: str,      # Colonna EMA Vol (da SPY_Volume)
                    threshold: float = 2.0) -> pd.Series:
        """
        Costruisce segnali discreti {-1, 0, +1} usando Z-score e filtro sui volumi.
        (Nessun costo di transazione applicato qui)
        """
        sig = pd.Series(0.0, index=df.index, dtype="float")

        sell_condition = (df[col_zs] > threshold) & (df[col_vol] < df[col_ema])
        buy_condition  = (df[col_zs] < -threshold) & (df[col_vol] > df[col_ema])

        sig[sell_condition] = -1.0
        sig[buy_condition]  =  1.0

        nan_mask = df[col_zs].isna() | df[col_vol].isna() | df[col_ema].isna()
        sig[nan_mask] = np.nan

        return sig

def add_features_and_signal(df_input: pd.DataFrame, 
                            ema_span_a: int,     # 'a' per EMA Volume
                            zscore_window_b: int, # 'b' per Z-Score (SMA e STDEV)
                            threshold: float = 2.0) -> pd.DataFrame:
    """
    Costruisce DataFrame con:
    - EMA dei volumi (span=a) su 'Volume'
    - Z-score dello spread (Window=b) su 'Spread'
    - Segnale discreto
    """
    df = df_input.copy()
    
    # 1. 'a' - Calcola EMA sul volume
    df['EMA_Vol'] = df['Volume'].ewm(span=ema_span_a, adjust=False).mean()
    
    # 2. 'b' - Calcola SMA sullo SPREAD
    df['SMA_Spread'] = df['Spread'].rolling(window=zscore_window_b).mean()
    
    # 3. 'b' - Calcola STDEV sullo SPREAD (usando la stessa finestra)
    df['STDEV_Spread'] = df['Spread'].rolling(window=zscore_window_b).std(ddof=1).replace(0, np.nan)
    
    # 4. Calcola Z-score
    df['Zscore'] = (df['Spread'] - df['SMA_Spread']) / df['STDEV_Spread']
    
    # 5. Costruisci il segnale
    df["Signal"]  = SignalBuilder.make_signal(
        df, "Zscore", "Volume", "EMA_Vol", threshold=threshold
    )

    # Rimuovi righe dove gli indicatori non sono ancora pronti
    return df.dropna(subset=["Zscore", "EMA_Vol", "Adj Close", "Volume"])

# =======================================================
# 3. Strategia state-machine (Vettorizzata)
# =======================================================

def implement_trading_strategy(df: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
    """
    Versione vettorizzata (senza loop 'for') della strategia di backtesting.
    """
    result = df.copy()
    if "Signal" not in result.columns:
        raise ValueError("df non contiene 'Signal'")

    # --- Setup Iniziale ---
    result["Position"] = result["Signal"].shift(1).fillna(0)
    prev_pos = result["Position"].shift(1).fillna(0)
    prev_close = result["Adj Close"].shift(1).fillna(0) 

    # --- 1. Vettorizzazione di "Entry_Price" ---
    trade_event = (result["Position"] != prev_pos)
    entry_px_signal = prev_close.where(trade_event & (result["Position"] != 0))
    result["Entry_Price"] = entry_px_signal.ffill().where(result["Position"] != 0, 0.0).fillna(0)

    # --- 2. Vettorizzazione di "Profit" (Trade Chiusi) ---
    prev_entry_price = result["Entry_Price"].shift(1).fillna(0)
    
    profit_long_trades = ((prev_close - prev_entry_price) / prev_entry_price * 100) * leverage
    profit_short_trades = ((prev_entry_price - prev_close) / prev_entry_price * 100) * leverage
    
    profit_long_trades = profit_long_trades.replace([np.inf, -np.inf], 0).fillna(0)
    profit_short_trades = profit_short_trades.replace([np.inf, -np.inf], 0).fillna(0)

    cond_long_exit = (prev_pos == 1) & (result["Position"] != 1)
    cond_short_exit = (prev_pos == -1) & (result["Position"] != -1)

    result["Profit"] = 0.0
    result["Profit"] = np.where(cond_long_exit, profit_long_trades, 0.0)
    result["Profit"] = np.where(cond_short_exit, profit_short_trades, result["Profit"])

    # --- 3. Vettorizzazione di "MToM" (Mark-to-Market Trade Aperti) ---
    mtom_long = ((result["Adj Close"] - result["Entry_Price"]) / result["Entry_Price"] * 100) * leverage
    mtom_short = ((result["Entry_Price"] - result["Adj Close"]) / result["Entry_Price"] * 100) * leverage
    
    result["MToM"] = 0.0
    result["MToM"] = np.where(result["Position"] == 1, mtom_long, 0.0)
    result["MToM"] = np.where(result["Position"] == -1, mtom_short, result["MToM"])
    result["MToM"] = result["MToM"].replace([np.inf, -np.inf], 0).fillna(0)

    # --- 4. Calcoli Finali Equity e Drawdown ---
    result["Cumulative_Profit"] = result["Profit"].cumsum()
    result["equity_curve"]      = result["Cumulative_Profit"] + result["MToM"] + 100 # Equity inizia a 100
    result["max_equity"]        = result["equity_curve"].cummax()
    
    # Drawdown in percentuale
    result["drawdown"] = (result["max_equity"] - result["equity_curve"]) / result["max_equity"] * 100
    result["max_drawdown"] = result["drawdown"].cummax()

    return result

# =========================
# 4. Metriche e Fitness
# =========================

def evaluate_performance(trade_df: pd.DataFrame, 
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calcola le metriche di performance chiave dal dataframe del backtest.
    """
    eq = trade_df["equity_curve"].values
    if len(eq) < 2:
        return {"max_drawdown": 0.0, "sharpe": 0.0, "total_return": 0.0, "trades_count": 0, "calmar_ratio": 0.0, "cagr": 0.0}

    # Calcolo Sharpe Ratio (sui ritorni % giornalieri dell'equity)
    returns_pct = pd.Series(eq).pct_change().fillna(0)
    mean_ret = returns_pct.mean()
    std_ret = returns_pct.std()
    sharpe = 0.0
    if std_ret > 1e-9:
        annualized_mean_ret = mean_ret * 252
        annualized_std = std_ret * np.sqrt(252)
        sharpe = (annualized_mean_ret - risk_free_rate) / annualized_std

    # Total Return (come % sull'equity iniziale di 100)
    total_return_pct = float(eq[-1] - 100.0)
    
    # Max Drawdown (già in % dal backtester)
    max_dd_pct = float(trade_df["max_drawdown"].iloc[-1]) if not trade_df["max_drawdown"].isna().all() else 0.0

    # Conteggio Trades
    trades_count = int((trade_df["Position"].diff().abs() > 0).sum() / 2) # /2 per contare entrate/uscite come 1 trade

    # CAGR (Compound Annual Growth Rate)
    num_days = len(eq) - 1
    num_years = num_days / 252.0
    cagr = 0.0
    if eq[0] > 0 and eq[-1] > 0 and num_years > 0:
        cagr = ((eq[-1] / eq[0]) ** (1.0 / num_years)) - 1.0
    
    # Calmar Ratio (CAGR / Max Drawdown %)
    calmar_ratio = 0.0
    if max_dd_pct > 1e-9:
        calmar_ratio = cagr / (max_dd_pct / 100.0) 
    elif cagr > 0:
        calmar_ratio = 1e9 # Ritorno positivo senza drawdown

    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct, 
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar_ratio, 
        "cagr_pct": cagr * 100,
        "trades_count": trades_count
    }

def fitness(df_data: pd.DataFrame,
            a: int, # Span EMA Volume
            b: int, # Window Z-Score (SMA & STDEV)
            leverage: float = 2.0,
            risk_free_rate: float = 0.02,
            min_sharpe: float = 0.0,
            min_return: float = 0.0) -> float:
    """
    Funzione di fitness per il GA.
    Obiettivo: Massimizzare il Calmar Ratio.
    """
    try:
        # 1. Costruisci segnale
        sig_df = add_features_and_signal(
            df_data, 
            ema_span_a=a, 
            zscore_window_b=b, 
            threshold=2.0
        )
        
        if sig_df.empty:
            return 1e9 # Penalità
            
        # 2. Esegui backtest
        trade_df = implement_trading_strategy(sig_df, leverage=leverage) 
        
        # 3. Valuta metriche
        m = evaluate_performance(trade_df, risk_free_rate)

        # 4. Applica vincoli
        if (m["trades_count"] < 10 or 
            m["sharpe_ratio"] < min_sharpe or 
            m["total_return_pct"] < min_return):
            return 1e9 # Penalità

        # 5. Ritorna il negativo del Calmar Ratio (perché il GA minimizza)
        return -m["calmar_ratio"]
    
    except Exception:
        return 1e9 # Penalità per qualsiasi errore

# =========================
# 5. Algoritmo Genetico (AB)
# =========================
class GeneticAlgorithmAB:
    def __init__(self,
                 population_size: int = 50,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 a_bounds: Tuple[int, int] = (5, 30),
                 b_bounds: Tuple[int, int] = (5, 30),
                 risk_free_rate: float = 0.02,
                 leverage: float = 2.0):
        
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.crossover_rate  = crossover_rate
        self.a_bounds        = a_bounds
        self.b_bounds        = b_bounds
        self.risk_free_rate  = risk_free_rate
        self.leverage        = leverage        

    def create_individual(self) -> Dict[str, int]:
        return {
            "a": random.randint(*self.a_bounds), # Per EMA Volume
            "b": random.randint(*self.b_bounds), # Per Z-Score Spread
        }

    def create_population(self) -> List[Dict[str, int]]:
        return [self.create_individual() for _ in range(self.population_size)]

    def crossover(self, p1: Dict[str, int], p2: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """ Crossover semplice """
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        
        c1 = {"a": p1["a"], "b": p2["b"]}
        c2 = {"a": p2["a"], "b": p1["b"]}
            
        return c1, c2

    def mutate(self, ind: Dict[str, int]) -> Dict[str, int]:
        out = ind.copy()
        if random.random() < self.mutation_rate:
            out["a"] += random.randint(-5, 5) # Mutazione finestra EMA
        if random.random() < self.mutation_rate:
            out["b"] += random.randint(-5, 5) # Mutazione finestra Z-Score
        
        # Applica i limiti (bounds)
        out["a"] = max(self.a_bounds[0], min(self.a_bounds[1], int(out["a"])))
        out["b"] = max(self.b_bounds[0], min(self.b_bounds[1], int(out["b"])))
        return out

    def run(self,
            df_data: pd.DataFrame,
            generations: int = 30) -> Tuple[Dict[str, int], Dict[str, float]]:
        
        population = self.create_population()
        best_ind = None
        best_score = float('inf')
        
        print(f"Avvio GA per {generations} generazioni (Pop: {self.population_size})...")

        for gen in range(generations):
            scores = []
            for ind in population:
                score = fitness(
                    df_data,
                    a=ind["a"], b=ind["b"],
                    leverage=self.leverage,
                    risk_free_rate=self.risk_free_rate,
                    min_sharpe=0.0, 
                    min_return=0.0
                )
                scores.append(score)

            ranked = sorted(zip(scores, population), key=lambda x: x[0])
            elites = [p for _, p in ranked[:self.population_size // 2]]

            if ranked[0][0] < best_score:
                best_score = ranked[0][0]
                best_ind = ranked[0][1]
            
            print(f"Gen {gen+1}/{generations} | Best Calmar (neg): {best_score:.4f} | Params (a, b): {best_ind}")

            offspring = []
            while len(offspring) < self.population_size // 2:
                p1, p2 = random.sample(elites, 2)
                c1, c2 = self.crossover(p1, p2)
                offspring.append(self.mutate(c1))
                if len(offspring) < self.population_size // 2:
                    offspring.append(self.mutate(c2))
            
            population = elites + offspring

        if best_ind is None:
             print("ERRORE: Ottimizzazione fallita.")
             return {"a": 0, "b": 0}, {}

        # Calcola metriche finali sui dati di training
        final_sig    = add_features_and_signal(
            df_data, 
            ema_span_a=best_ind["a"], 
            zscore_window_b=best_ind["b"]
        )
        final_trades = implement_trading_strategy(final_sig, leverage=self.leverage)
        final_metrics = evaluate_performance(final_trades, self.risk_free_rate)
        
        return best_ind, final_metrics

# =========================
# 6. Esecuzione Principale
# =========================

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42) 

    # --- Parametri Globali ---
    START_DATE = "2007-01-01"
    SPLIT_DATE = "2022-01-01"
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    LEVERAGE = 2.0
    RISK_FREE_RATE = 0.02 # 2% come richiesto
    
    # --- FASE 1: Caricamento Dati ---
    df_train, df_test = load_and_prepare_data(START_DATE, END_DATE, SPLIT_DATE)

    # --- FASE 2: OTTIMIZZAZIONE (Solo su Training Set) ---
    print("\n--- FASE 1: Avvio Ottimizzazione (Training Set) ---")
    
    ga = GeneticAlgorithmAB( # <-- Usa la classe AB
        population_size=50,   
        mutation_rate=0.2,
        crossover_rate=0.8,
        a_bounds=(5, 30),    # Limiti per 'a' (EMA Vol)
        b_bounds=(5, 30),    # Limiti per 'b' (Z-Score Spread)
        risk_free_rate=RISK_FREE_RATE,
        leverage=LEVERAGE
    )

    # Esegui il GA (passa solo dati di TRAIN)
    best_params, train_metrics = ga.run(
        df_train, 
        generations=30 # Numero di generazioni
    )

    print("\n--- Ottimizzazione (Training Set) Completata ---")
    print(f"Parametri ottimali trovati:")
    print(f"  a (Span EMA Volume SPY): {best_params['a']}")
    print(f"  b (Window Z-Score Spread): {best_params['b']}")
    
    print("\nMetriche finali sul Training Set (In-Sample):")
    for key, value in train_metrics.items():
        print(f"   - {key}: {value:.4f}")

    
    # --- FASE 3: BACKTEST (Solo su Test Set) ---
    print(f"\n--- FASE 2: Avvio Backtest (Test Set 'Out-of-Sample') ---")
    print(f"Applicazione parametri fissi: a={best_params['a']}, b={best_params['b']}, leva={LEVERAGE}x")

    # 1. Costruisci il segnale sul set di test
    test_sig_df = add_features_and_signal(
        df_test, 
        ema_span_a=best_params['a'], 
        zscore_window_b=best_params['b'], 
        threshold=2.0
    )

    if test_sig_df.empty:
        print("ERRORE: Nessun segnale generato sul Test Set.")
    else:
        # 2. Esegui la strategia sul set di test
        test_trade_df = implement_trading_strategy(
            test_sig_df, 
            leverage=LEVERAGE
        )
        
        # 3. Valuta le performance sul set di test
        test_metrics = evaluate_performance(
            test_trade_df, 
            risk_free_rate=RISK_FREE_RATE
        )
        
        print("\n--- Risultati Backtest (Test Set 'Out-of-Sample') ---")
        for key, value in test_metrics.items():
            print(f"   - {key}: {value:.4f}")
            
        # --- FASE 4: Visualizzazione ---
        print("\nGenerazione grafico equity line...")
        
        train_sig_df = add_features_and_signal(
            df_train, 
            ema_span_a=best_params['a'], 
            zscore_window_b=best_params['b']
        )
        
        train_trade_df = implement_trading_strategy(train_sig_df, leverage=LEVERAGE)
        
        plt.figure(figsize=(14, 7))
        plt.plot(train_trade_df.index, train_trade_df['equity_curve'], label='Training (In-Sample)', color='blue')
        plt.plot(test_trade_df.index, test_trade_df['equity_curve'], label='Test (Out-of-Sample)', color='orange')
        
        plt.axvline(pd.to_datetime(SPLIT_DATE), color='grey', linestyle='--', label=f'Split Date ({SPLIT_DATE})')
        
        plt.title(f"Equity Curve - Z-Score Spread (Leva {LEVERAGE}x)\nParams (a={best_params['a']}, b={best_params['b']})", fontsize=14)
        plt.ylabel('Equity')
        plt.xlabel('Data')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Formattazione asse X
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig("equity_curve_AB.png")
        print("Grafico salvato come 'equity_curve_AB.png'")
        # plt.show()