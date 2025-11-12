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
                    col_zs: str,
                    col_vol: str,
                    col_ema: str,
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
                 a: int,
                 b: int,
                 threshold: float = 2.0) -> pd.DataFrame:
    """
    Costruisce DataFrame con:
    - EMA dei volumi (span=a)
    - Z-score del prezzo con finestra b
    - Segnale discreto via SignalBuilder
    """
    df = pd.DataFrame({"Close": close, "Volume": volume})
    ema_vol = calculate_ema(df["Volume"], span=a)
    zscore  = calculate_zscore(df["Close"], window=b)

    df["EMA_Vol"] = ema_vol
    df["Zscore"]  = zscore
    df["Signal"]  = SignalBuilder.make_signal(df, "Zscore", "Volume", "EMA_Vol", threshold=2.0)

    return df.dropna(subset=["Zscore", "EMA_Vol"])

# =========================
# Strategia state-machine
# =========================
def implement_trading_strategy(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Signal" not in result.columns:
        raise ValueError("df non contiene 'Signal'")

    result["Position"]     = result["Signal"].shift(1).fillna(0)
    result["Entry_Price"]  = 0.0
    result["Profit"]       = 0.0
    result["MToM"]         = 0.0

    for i in range(1, len(result)):
        px        = result.iloc[i]["Close"]
        prev_px   = result.iloc[i-1]["Close"]
        pos       = result.iloc[i]["Position"]
        prev_pos  = result.iloc[i-1]["Position"]

        if pos == 1:  # long
            if prev_pos == 0:
                result.at[i, "Entry_Price"] = prev_px
            if prev_pos == -1:
                result.at[i, "Profit"] = (result.at[i-1, "Entry_Price"] - prev_px) / result.at[i-1, "Entry_Price"] * 100
                result.at[i, "Entry_Price"] = prev_px
            if prev_pos == 1:
                result.at[i, "Entry_Price"] = result.at[i-1, "Entry_Price"]
            result.at[i, "MToM"] = (px - result.at[i, "Entry_Price"]) / result.at[i, "Entry_Price"] * 100

        elif pos == -1:  # short
            if prev_pos == 0:
                result.at[i, "Entry_Price"] = prev_px
            if prev_pos == 1:
                result.at[i, "Profit"] = (prev_px - result.at[i-1, "Entry_Price"]) / result.at[i-1, "Entry_Price"] * 100
                result.at[i, "Entry_Price"] = prev_px
            if prev_pos == -1:
                result.at[i, "Entry_Price"] = result.at[i-1, "Entry_Price"]
            result.at[i, "MToM"] = (result.at[i, "Entry_Price"] - px) / result.at[i, "Entry_Price"] * 100

        else:  # flat
            if prev_pos == -1:
                result.at[i, "Profit"] = (result.at[i-1, "Entry_Price"] - prev_px) / result.at[i-1, "Entry_Price"] * 100
            if prev_pos == 1:
                result.at[i, "Profit"] = (prev_px - result.at[i-1, "Entry_Price"]) / result.at[i-1, "Entry_Price"] * 100

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
        return {"max_drawdown": 0.0, "sharpe": 0.0, "total_return": 0.0, "trades_count": 0}

    daily = np.diff(eq)
    vol = daily.std()
    sharpe = 0.0 if vol == 0 else (daily.mean() / vol) * np.sqrt(252)
    total_return = float(eq[-1] - eq[0])
    max_dd = float(trade_df["max_drawdown"].iloc[-1]) if not trade_df["max_drawdown"].isna().all() else 0.0
    trades_count = int((trade_df["Position"].diff().abs() > 0).sum())

    return {
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_return": total_return,
        "trades_count": trades_count
    }

def fitness(close: pd.Series,
            volume: pd.Series,
            a: int,
            b: int,
            min_sharpe: float = 0.0,
            min_return: float = 0.0) -> float:
    sig_df = build_signal(close, volume, a=a, b=b, threshold=2.0)
    if sig_df.empty:
        return 1e9
    trade_df = implement_trading_strategy(sig_df)
    m = evaluate_performance(trade_df)

    if m["trades_count"] == 0 or m["sharpe"] < min_sharpe or m["total_return"] < min_return:
        return 1e9

    return m["max_drawdown"]

# =========================
# Algoritmo genetico
# =========================
class GeneticAlgorithmAB:
    def __init__(self,
                 population_size: int = 20,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 a_bounds: Tuple[int, int] = (5, 50),
                 b_bounds: Tuple[int, int] = (5, 50),
                 min_sharpe: float = 0.0,
                 min_return: float = 0.0):
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.crossover_rate  = crossover_rate
        self.a_bounds        = a_bounds
        self.b_bounds        = b_bounds
        self.min_sharpe      = min_sharpe
        self.min_return      = min_return

    def create_individual(self) -> Dict[str, int]:
        return {
            "a": random.randint(*self.a_bounds),
            "b": random.randint(*self.b_bounds)
        }

    def create_population(self) -> List[Dict[str, int]]:
        return [self.create_individual() for _ in range(self.population_size)]

    def crossover(self, p1: Dict[str, int], p2: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        # single-point crossover: scambia b
        c1 = {"a": p1["a"], "b": p2["b"]}
        c2 = {"a": p2["a"], "b": p1["b"]}
        return c1, c2

    def mutate(self, ind: Dict[str, int]) -> Dict[str, int]:
        out = ind.copy()
        if random.random() < self.mutation_rate:
            out["a"] += random.randint(-5, 5)
        if random.random() < self.mutation_rate:
            out["b"] += random.randint(-5, 5)
        # clamp bounds
        out["a"] = max(self.a_bounds[0], min(self.a_bounds[1], int(out["a"])))
        out["b"] = max(self.b_bounds[0], min(self.b_bounds[1], int(out["b"])))
        return out

    def run(self,
            close: pd.Series,
            volume: pd.Series,
            generations: int = 50) -> Tuple[Dict[str, int], Dict[str, float]]:
        population = self.create_population()
        best_ind = None
        best_score = float('inf')

        for gen in range(generations):
            scores = []
            for ind in population:
                score = fitness(close, volume,
                                a=ind["a"], b=ind["b"],
                                min_sharpe=self.min_sharpe,
                                min_return=self.min_return)
                scores.append(score)

            # selezione (top 50%)
            ranked = sorted(zip(scores, population), key=lambda x: x[0])
            elites = [p for _, p in ranked[:self.population_size // 2]]

            # aggiorna best
            if ranked[0][0] < best_score:
                best_score = ranked[0][0]
                best_ind = ranked[0][1]

            # genera offspring
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
        final_sig    = build_signal(close, volume, a=best_ind["a"], b=best_ind["b"])
        final_trades = implement_trading_strategy(final_sig)
        final_metrics = evaluate_performance(final_trades)
        return best_ind, final_metrics


# =========================
# Esempio d'uso
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    # Simula dati: prezzi e volumi
    n_days = 500
    close = pd.Series(100 + np.cumsum(np.random.normal(0, 1, n_days)))
    volume = pd.Series(1000 + np.random.normal(0, 100, n_days)).abs()

    # Inizializza GA
    ga = GeneticAlgorithmAB(
        population_size=10,
        mutation_rate=0.2,
        crossover_rate=0.8,
        a_bounds=(5, 30),
        b_bounds=(5, 30),
        min_sharpe=0.0,
        min_return=0.0
    )

    best_params, metrics = ga.run(close, volume, generations=20)
    print("Parametri ottimali:", best_params)
    print("Metriche:", metrics)