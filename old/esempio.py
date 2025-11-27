import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple

# =========================
# Utils: EMA e Z-score
# =========================
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calcola EMA su una serie di dati."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calcola Z-score su una serie di dati."""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    std = std.replace(0, np.nan)
    return (series - ma) / std


# =========================
# Signal builder
# =========================
class SignalBuilder:
    @staticmethod
    def make_signal(df: pd.DataFrame,
                    col_zs: str,       # Colonna Z-score (da DJ_SPREAD)
                    col_vol: str,      # Colonna Volume (da SPY_Volume)
                    col_ema: str,      # Colonna EMA Vol (da SPY_Volume)
                    threshold: float = 2.0) -> pd.Series:
        """
        Costruisce segnali discreti {-1, 0, +1} usando Z-score e filtro sui volumi.
        - SELL (-1): Zscore > threshold e Volumi < EMA_volumi
        - BUY (+1): Zscore < -threshold e Volumi > EMA_volumi
        - FLAT (0): altrimenti
        """
        sig = pd.Series(0.0, index=df.index, dtype="float")

        sell_condition = (df[col_zs] > threshold) & (df[col_vol] < df[col_ema])
        buy_condition  = (df[col_zs] < -threshold) & (df[col_vol] > df[col_ema])

        sig[sell_condition] = -1.0
        sig[buy_condition]  =  1.0

        nan_mask = df[col_zs].isna() | df[col_vol].isna() | df[col_ema].isna()
        sig[nan_mask] = np.nan

        return sig

# =========================
# Costruzione del segnale
# =========================
def build_signal(close: pd.Series,
                 volume: pd.Series,
                 spread: pd.Series,   # <--- AGGIUNTA SERIE SPREAD
                 a: int,
                 b: int,
                 threshold: float = 2.0) -> pd.DataFrame:
    """
    Costruisce DataFrame con:
    - EMA dei volumi (span=a) su 'volume'
    - Z-score dello spread (finestra b) su 'spread'
    - Segnale discreto via SignalBuilder
    """
    
    # 1. Calcola EMA sul volume (usa 'a')
    ema_vol = calculate_ema(volume, span=a)
    
    # 2. Calcola Z-score sullo SPREAD (usa 'b')
    zscore  = calculate_zscore(spread, window=b)

    # 3. Assembla il DataFrame per il backtest
    df = pd.DataFrame({
        "Adj Close": close,   
        "Volume": volume,
        "EMA_Vol": ema_vol,
        "Zscore": zscore
    })
    
    # 4. Costruisci il segnale
    df["Signal"]  = SignalBuilder.make_signal(df, "Zscore", "Volume", "EMA_Vol", threshold=threshold)

    return df.dropna(subset=["Zscore", "EMA_Vol", "Adj Close", "Volume"])

# =======================================================
# Strategia state-machine (Versione Vettorizzata Veloce)
# =======================================================
def implement_trading_strategy(df: pd.DataFrame, leverage: float = 2.0) -> pd.DataFrame: # <--- MODIFICA LEVA
    """
    Versione vettorizzata (senza loop 'for') della strategia di backtesting.
    """
    result = df.copy()
    if "Signal" not in result.columns:
        raise ValueError("df non contiene 'Signal'")

    # --- Setup Iniziale ---
    result["Position"] = result["Signal"].shift(1).fillna(0)
    prev_pos = result["Position"].shift(1).fillna(0)
    prev_close = result["Adj Close"].shift(1).fillna(0) # Usa il 'Close' di SPY

    # --- 1. Vettorizzazione di "Entry_Price" ---
    trade_event = (result["Position"] != prev_pos)
    entry_px_signal = prev_close.where(trade_event & (result["Position"] != 0))
    result["Entry_Price"] = entry_px_signal.ffill().where(result["Position"] != 0, 0.0).fillna(0)

    # --- 2. Vettorizzazione di "Profit" ---
    prev_entry_price = result["Entry_Price"].shift(1).fillna(0)
    
    # I profitti sono calcolati sul 'Close' di SPY
    profit_long_trades = ((prev_close - prev_entry_price) / prev_entry_price * 100) * leverage # <--- USA LEVA
    profit_short_trades = ((prev_entry_price - prev_close) / prev_entry_price * 100) * leverage # <--- USA LEVA
    
    profit_long_trades = profit_long_trades.replace([np.inf, -np.inf], 0).fillna(0)
    profit_short_trades = profit_short_trades.replace([np.inf, -np.inf], 0).fillna(0)

    cond_long_exit = (prev_pos == 1) & (result["Position"] != 1)
    cond_short_exit = (prev_pos == -1) & (result["Position"] != -1)

    result["Profit"] = 0.0
    result["Profit"] = np.where(cond_long_exit, profit_long_trades, 0.0)
    result["Profit"] = np.where(cond_short_exit, profit_short_trades, result["Profit"])

    # --- 3. Vettorizzazione di "MToM" ---
    # Il MToM è calcolato sul 'Close' di SPY
    mtom_long = ((result["Adj Close"] - result["Entry_Price"]) / result["Entry_Price"] * 100) * leverage # <--- USA LEVA
    mtom_short = ((result["Entry_Price"] - result["Adj Close"]) / result["Entry_Price"] * 100) * leverage # <--- USA LEVA
    
    result["MToM"] = 0.0
    result["MToM"] = np.where(result["Position"] == 1, mtom_long, 0.0)
    result["MToM"] = np.where(result["Position"] == -1, mtom_short, result["MToM"])
    result["MToM"] = result["MToM"].replace([np.inf, -np.inf], 0).fillna(0)

    # --- 4. Calcoli Finali ---
    result["Cumulative_Profit"] = result["Profit"].cumsum()
    result["equity_curve"]      = result["Cumulative_Profit"] + result["MToM"] + 100
    result["max_equity"]        = result["equity_curve"].cummax()
    result["drawdown"]          = result["max_equity"] - result["equity_curve"]
    result["max_drawdown"]      = result["drawdown"].cummax()
    result["drawdown_length"]   = (result["drawdown"] > 0).astype(int).groupby(result["drawdown"].le(0).cumsum()).cumsum()

    return result

# =========================
# Metriche e fitness
# =========================
def evaluate_performance(trade_df: pd.DataFrame) -> Dict[str, float]:
    eq = trade_df["equity_curve"].values
    if len(eq) < 2:
        return {"max_drawdown": 0.0, "sharpe": 0.0, "total_return": 0.0, "trades_count": 0, "calmar_ratio": 0.0, "cagr": 0.0}

    daily = np.diff(eq)
    vol = daily.std()
    sharpe = 0.0 if vol == 0 else (daily.mean() / vol) * np.sqrt(252)
    total_return = float(eq[-1] - eq[0])
    max_dd = float(trade_df["max_drawdown"].iloc[-1]) if not trade_df["max_drawdown"].isna().all() else 0.0
    trades_count = int((trade_df["Position"].diff().abs() > 0).sum())
    
    roll_max = np.maximum.accumulate(eq)
    drawdown = (eq - roll_max) / roll_max 
    max_dd_pct = np.max(-drawdown) 

    num_days = len(eq) - 1
    num_years = num_days / 252.0
    if eq[0] <= 0 or eq[-1] <= 0 or num_years <= 0:
        cagr = 0.0
    else:
        cagr = ((eq[-1] / eq[0]) ** (1.0 / num_years)) - 1.0
    
    if max_dd_pct is None or max_dd_pct <= 1e-9:
        calmar_ratio = 1e9 if cagr > 0 else 0.0 
    else:
        calmar_ratio = cagr / max_dd_pct

    return {
        "max_drawdown": max_dd, 
        "sharpe": sharpe,
        "total_return": total_return,
        "trades_count": trades_count,
        "calmar_ratio": calmar_ratio, 
        "cagr": cagr
    }

# --- FITNESS (MODIFICATA) ---
def fitness(close: pd.Series,
            volume: pd.Series,
            spread: pd.Series,   # <--- AGGIUNTO
            a: int,
            b: int,
            leverage: float = 2.0,   # <--- MODIFICA LEVA
            min_sharpe: float = 0.0,
            min_return: float = 0.0) -> float:
    
    # Passa tutte e 3 le serie
    sig_df = build_signal(close, volume, spread, a=a, b=b, threshold=2.0)
    
    if sig_df.empty:
        return 1e9
        
    trade_df = implement_trading_strategy(sig_df, leverage=leverage) 
    m = evaluate_performance(trade_df)

    if m["trades_count"] == 0 or m["sharpe"] < min_sharpe or m["total_return"] < min_return:
        return 1e9

    return -m["calmar_ratio"]

# =========================
# Algoritmo genetico
# =========================
class GeneticAlgorithmAB:
    def __init__(self,
                 population_size: int = 50,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 a_bounds: Tuple[int, int] = (5, 50),
                 b_bounds: Tuple[int, int] = (5, 50),
                 min_sharpe: float = 0.0,
                 min_return: float = 0.0,
                 leverage: float = 2.0): # <--- MODIFICA LEVA
        
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.crossover_rate  = crossover_rate
        self.a_bounds        = a_bounds
        self.b_bounds        = b_bounds
        self.min_sharpe      = min_sharpe
        self.min_return      = min_return
        self.leverage        = leverage        

    def create_individual(self) -> Dict[str, int]:
        return {
            "a": random.randint(*self.a_bounds), # Per EMA Volume
            "b": random.randint(*self.b_bounds)  # Per Z-score Spread
        }

    def create_population(self) -> List[Dict[str, int]]:
        return [self.create_individual() for _ in range(self.population_size)]

    def crossover(self, p1: Dict[str, int], p2: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        c1 = {"a": p1["a"], "b": p2["b"]}
        c2 = {"a": p2["a"], "b": p1["b"]}
        return c1, c2

    def mutate(self, ind: Dict[str, int]) -> Dict[str, int]:
        out = ind.copy()
        if random.random() < self.mutation_rate:
            out["a"] += random.randint(-5, 5)
        if random.random() < self.mutation_rate:
            out["b"] += random.randint(-5, 5)
        out["a"] = max(self.a_bounds[0], min(self.a_bounds[1], int(out["a"])))
        out["b"] = max(self.b_bounds[0], min(self.b_bounds[1], int(out["b"])))
        return out

    # --- RUN (MODIFICATA) ---
    def run(self,
            close: pd.Series,
            volume: pd.Series,
            spread: pd.Series,   # <--- AGGIUNTO
            generations: int = 30) -> Tuple[Dict[str, int], Dict[str, float]]:
        
        population = self.create_population()
        best_ind = None
        best_score = float('inf')

        for gen in range(generations):
            scores = []
            for ind in population:
                # Passa tutte e 3 le serie a fitness
                score = fitness(close, volume, spread,
                                a=ind["a"], b=ind["b"],
                                leverage=self.leverage,
                                min_sharpe=self.min_sharpe,
                                min_return=self.min_return)
                scores.append(score)

            ranked = sorted(zip(scores, population), key=lambda x: x[0])
            elites = [p for _, p in ranked[:self.population_size // 2]]

            if ranked[0][0] < best_score:
                best_score = ranked[0][0]
                best_ind = ranked[0][1]

            offspring = []
            while len(offspring) < self.population_size // 2:
                p1, p2 = random.sample(elites, 2)
                c1, c2 = self.crossover(p1, p2)
                offspring.append(self.mutate(c1))
                if len(offspring) < self.population_size // 2:
                    offspring.append(self.mutate(c2))
            population = elites + offspring

        if best_ind is None:
            best_ind = ranked[0][1]

        # metriche finali
        final_sig    = build_signal(close, volume, spread, a=best_ind["a"], b=best_ind["b"])
        final_trades = implement_trading_strategy(final_sig, leverage=self.leverage)
        final_metrics = evaluate_performance(final_trades)
        
        return best_ind, final_metrics

# =========================
# Funzione Helper Caricamento
# =========================
def load_data_series(file_path: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Carica e prepara le serie da un file Excel."""
    print(f"\nCaricamento dati da: {file_path}")
    try:
        data = pd.read_excel(
            file_path, 
            parse_dates=True, 
            index_col='Date'
        )

        # --- ESTRAZIONE DELLE 3 SERIE ---
        close_series = data["SPY"]
        volume_series = data["SPY_Volume"]
        spread_series = data["DJ_SPREAD"] 
        
        # Allinea gli indici per sicurezza
        common_index = close_series.index.intersection(volume_series.index).intersection(spread_series.index)
        close_series = close_series.loc[common_index]
        volume_series = volume_series.loc[common_index]
        spread_series = spread_series.loc[common_index]
        
        if close_series.empty or volume_series.empty or spread_series.empty:
            raise ValueError("Una delle serie è vuota dopo l'allineamento.")
        
        print(f"Dati caricati con successo: {len(close_series)} giorni.")
        return close_series, volume_series, spread_series

    except FileNotFoundError:
        print(f"ERRORE: File non trovato a {file_path}")
        exit()
    except KeyError as e:
        print(f"ERRORE: La colonna {e} non è stata trovata nel file.")
        print("Assicurati che 'SPY', 'SPY_Volume' e 'DJ_SPREAD' siano presenti.")
        exit()
    except Exception as e:
        print(f"ERRORE generico durante il caricamento: {e}")
        exit()

# ===============================================
# Esecuzione con Dati Reali (TRAIN e TEST)
# ===============================================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42) 

    # --- Definizioni File ---
    train_file_path = "analisi_tecnica_train.xlsx"
    test_file_path  = "analisi_tecnica_test.xlsx"
    LEVERAGE_SETTING = 2.0
    
    # --- FASE 1: OTTIMIZZAZIONE (Solo su Training Set) ---
    print("--- FASE 1: Avvio Ottimizzazione (Training Set) ---")
    
    # Carica i dati di training
    train_close, train_volume, train_spread = load_data_series(train_file_path)

    # Inizializza l'Algoritmo Genetico
    ga = GeneticAlgorithmAB(
        population_size=50,   
        mutation_rate=0.2,
        crossover_rate=0.8,
        a_bounds=(5, 30),     # Limiti per span EMA Volume
        b_bounds=(5, 30),     # Limiti per finestra Z-score Spread
        min_sharpe=0.1,       
        min_return=0.0,
        leverage=LEVERAGE_SETTING
    )

    # Esegui il GA (passa solo dati di TRAIN)
    best_params, train_metrics = ga.run(
        train_close, 
        train_volume, 
        train_spread,
        generations=30
    )

    print("\n--- Ottimizzazione (Training Set) Completata ---")
    print(f"Parametri ottimali trovati:")
    print(f"  a (Span EMA Volume SPY): {best_params['a']}")
    print(f"  b (Window Z-score DJ_SPREAD): {best_params['b']}")
    
    print("\nMetriche finali sul Training Set (In-Sample):")
    for key, value in train_metrics.items():
        print(f"   - {key}: {value:.4f}")

    
    # --- FASE 2: BACKTEST (Solo su Test Set) ---
    print("\n--- FASE 2: Avvio Backtest (Test Set 'Out-of-Sample') ---")
    
    # Carica i dati di test
    test_close, test_volume, test_spread = load_data_series(test_file_path)

    print(f"\nApplicazione parametri fissi: a={best_params['a']}, b={best_params['b']}")

    # 1. Costruisci il segnale sul set di test
    test_sig_df = build_signal(
        test_close, 
        test_volume, 
        test_spread, 
        a=best_params['a'], 
        b=best_params['b'], 
        threshold=2.0
    )

    if test_sig_df.empty:
        print("ERRORE: Nessun segnale generato sul Test Set. Impossibile continuare.")
    else:
        # 2. Esegui la strategia sul set di test
        test_trade_df = implement_trading_strategy(
            test_sig_df, 
            leverage=LEVERAGE_SETTING
        )
        
        # 3. Valuta le performance sul set di test
        test_metrics = evaluate_performance(test_trade_df)
        
        print("\n--- Risultati Backtest (Test Set 'Out-of-Sample') ---")
        print("\nMetriche finali sul Test Set:")
        for key, value in test_metrics.items():
            print(f"   - {key}: {value:.4f}")