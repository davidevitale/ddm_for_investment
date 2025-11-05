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
        ema_tickers = ['SPY', 'QQQ']
        for ticker in ema_tickers:
            if ticker in self.analyzer.normalized_prices.columns:
                norm_price = self.analyzer.normalized_prices[ticker]
                self.results['ema'][f"{ticker}_EMA_20"] = self.calculate_ema(norm_price, a=20)
            else:
                print(f"    Warning: Ticker {ticker} non trovato nei prezzi normalizzati per EMA.")

        # 3. Calcolo Z-Score
        print("  Calcolo Z-Score (b=20) per 'DJ_SPREAD'...")
        self.results['zscore'] = {}
        if 'DJ_SPREAD' in self.analyzer.price_data.columns:
            spread_data = self.analyzer.price_data['DJ_SPREAD']
            self.results['zscore']['DJ_SPREAD_ZScore_20'] = self.calculate_zscore(spread_data, b=20)
        else:
            print("    Warning: Colonna 'DJ_SPREAD' non trovata per calcolo Z-Score.")
            
        print("\nAnalisi tecnica completata. I risultati sono in 'tech_analyzer.results'")


