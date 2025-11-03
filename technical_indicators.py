from datetime import datetime
from data import MarketAnalyzer

class TechnicalAnalyzer:
    """
    Una classe per eseguire calcoli di analisi tecnica sui dati
    di un oggetto MarketAnalyzer.
    """
    
    def __init__(self, market_analyzer):
        """
        Inizializza l'analizzatore tecnico.
        
        Args:
            market_analyzer (MarketAnalyzer): Un'istanza di MarketAnalyzer
                                              che ha già eseguito .run_analysis()
        """
        self.analyzer = market_analyzer
        self.results = {} # Dizionario per conservare i risultati
        
        if self.analyzer.price_data is None:
            print("Errore: L'oggetto MarketAnalyzer non ha dati (price_data is None).")
            print("Eseguire prima .run_analysis() sull'oggetto MarketAnalyzer.")
            return

        if self.analyzer.normalized_prices is None:
            print("Errore: L'oggetto MarketAnalyzer non ha dati (normalized_prices is None).")
            print("Eseguire prima .run_analysis() sull'oggetto MarketAnalyzer.")
            return
            
        print(f"\nTechnicalAnalyzer initialized.")
        print("Pronto per calcolare EMA, MACD e Z-Score.")
    
    def calculate_ema(self, data_series, a=20):
        """
        Calcola la Media Mobile Esponenziale (EMA) per una serie di dati.
        
        Args:
            data_series (pd.Series): La serie di prezzi.
            a (int): Il numero di periodi per l'EMA. Default 20.
        
        Returns:
            pd.Series: La serie di dati EMA.
        """
        # adjust=False è comunemente usato in TA per far coincidere 
        # le formule con i software di trading.
        return data_series.ewm(span=a, adjust=False).mean()

    def calculate_macd(self, data_series, k_fast=12, k_slow=26, k_signal=9):
        """
        Calcola il MACD (Line, Signal, Histogram) per una serie di dati.
        
        Args:
            data_series (pd.Series): La serie di prezzi.
            k_fast (int): Periodo EMA veloce (default 12).
            k_slow (int): Periodo EMA lento (default 26).
            k_signal (int): Periodo EMA per la linea di segnale (default 9).
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        # ● MACD Line: EMA(k=12) - EMA(k=26)
        ema_fast = self.calculate_ema(data_series, a=k_fast)
        ema_slow = self.calculate_ema(data_series, a=k_slow)
        macd_line = ema_fast - ema_slow
        
        # ● Signal Line: k=9 EMA of the MACD Line
        signal_line = self.calculate_ema(macd_line, a=k_signal)
        
        # ● MACD Histogram: MACD Line - Signal Line
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_zscore(self, data_series, b=20):
        """
        Calcola lo Z-Score di una serie di dati basato su una media mobile.
        
        Args:
            data_series (pd.Series): La serie di dati (es. lo spread).
            b (int): Il numero di periodi per la media/dev.std mobile. Default 20.
            
        Returns:
            pd.Series: La serie di dati Z-Score.
        """
        # Calcola la media mobile
        moving_avg = data_series.rolling(window=b).mean()
        
        # Calcola la deviazione standard mobile
        moving_std = data_series.rolling(window=b).std()
        
        # Calcola lo Z-Score
        # Z-Score = (Valore Attuale - Media Mobile) / Deviazione Standard Mobile
        z_score = (data_series - moving_avg) / moving_std
        
        return z_score

    def run_technical_analysis(self):
        """
        Esegue tutti i calcoli tecnici richiesti e salva i risultati.
        """
        print("\nAvvio calcoli di analisi tecnica...")
        
        # 1. Calcolo EMA
        print("  Calcolo EMA (a=20) per 'DXY', 'SPY', 'QQQ', 'GLD'...")
        self.results['ema'] = {}
        ema_tickers = ['DX-Y.NYB', 'SPY', 'QQQ', 'GLD']
        for ticker in ema_tickers:
            if ticker in self.analyzer.normalized_prices.columns:
                norm_price = self.analyzer.normalized_prices[ticker]
                self.results['ema'][f"{ticker}_EMA_20"] = self.calculate_ema(norm_price, a=20)
            else:
                print(f"    Warning: Ticker {ticker} non trovato nei prezzi normalizzati per EMA.")

        # 2. Calcolo MACD
        print("  Calcolo MACD per 'SPY' e 'QQQ' (su prezzi normalizzati)...")
        self.results['macd'] = {}
        macd_tickers = ['SPY', 'QQQ']
        for ticker in macd_tickers:
            if ticker in self.analyzer.normalized_prices.columns:
                norm_price = self.analyzer.normalized_prices[ticker]
                macd_l, signal_l, histo = self.calculate_macd(norm_price)
                self.results['macd'][f"{ticker}_MACD_Line"] = macd_l
                self.results['macd'][f"{ticker}_Signal_Line"] = signal_l
                self.results['macd'][f"{ticker}_Histogram"] = histo
            else:
                print(f"    Warning: Ticker {ticker} non trovato nei prezzi normalizzati per MACD.")

        # 3. Calcolo Z-Score
        print("  Calcolo Z-Score (b=20) per 'DJ_SPREAD'...")
        self.results['zscore'] = {}
        if 'DJ_SPREAD' in self.analyzer.price_data.columns:
            spread_data = self.analyzer.price_data['DJ_SPREAD']
            self.results['zscore']['DJ_SPREAD_ZScore_20'] = self.calculate_zscore(spread_data, b=20)
        else:
            print("    Warning: Colonna 'DJ_SPREAD' non trovata per calcolo Z-Score.")
            
        print("\nAnalisi tecnica completata. I risultati sono in 'tech_analyzer.results'")


# --- ESECUZIONE DELLO SCRIPT ---

# 1. Definisci l'universo di investimento
new_data = {
    'SPY': 'S&P 500 ETF (Adj.)',
    'QQQ': 'NASDAQ 100 ETF (Adj.)',
    'GLD': 'Gold ETF',
    'DX-Y.NYB': 'US Dollar Index (DXY)',
    '^VIX': 'CBOE Volatility Index (VIX)',
    '^VVIX': 'CBOE VIX Volatility Index (VVIX)',
    '^DJI': 'Dow Jones Industrial Average (DJIA)',
    '^DJT': 'Dow Jones Transportation Average (DJT)'
}

# 2. Imposta il periodo di analisi
start_date = '2007-04-10'
end_date = datetime.now().strftime('%Y-%m-%d')

# 3. Crea l'istanza e avvia l'analisi (CLASSE 1)
try:
    analyzer = MarketAnalyzer(tickers_dict=new_data, start_date=start_date, end_date=end_date)
    analyzer.run_analysis()

    # --- NUOVA SEZIONE ---
    # 4. Crea l'istanza e avvia l'analisi (CLASSE 2)
    
    # Controlla se l'analisi precedente è andata a buon fine
    if analyzer.price_data is not None and not analyzer.price_data.empty:
        
        tech_analyzer = TechnicalAnalyzer(analyzer)
        tech_analyzer.run_technical_analysis()
        
        # Stampa alcuni risultati per verifica
        print("\n--- Esempio Risultati (ultime 5 righe) ---")
        
        # MACD SPY
        if 'SPY_Histogram' in tech_analyzer.results['macd']:
            print("\nMACD (SPY):")
            print(tech_analyzer.results['macd']['SPY_Histogram'].tail())
            
        # Z-Score
        if 'DJ_SPREAD_ZScore_20' in tech_analyzer.results['zscore']:
            print("\nZ-Score (DJ_SPREAD):")
            print(tech_analyzer.results['zscore']['DJ_SPREAD_ZScore_20'].tail())
            
        # EMA ORO
        if 'GLD_EMA_20' in tech_analyzer.results['ema']:
            print("\nEMA (GLD, a=20):")
            print(tech_analyzer.results['ema']['GLD_EMA_20'].tail())

    else:
        print("\nAnalisi tecnica non eseguita perché l'analisi di mercato iniziale è fallita.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")