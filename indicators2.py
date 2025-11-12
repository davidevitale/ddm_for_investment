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
        
        if self.analyzer.data is None:
            print("Errore: L'oggetto MarketAnalyzer non ha dati (self.analyzer.data is None).")
            print("Eseguire prima .run_analysis() sull'oggetto MarketAnalyzer.")
            return
            
        print(f"\nTechnicalAnalyzer initialized.")
        print("Pronto per calcolare EMA (Volumi) e Z-Score (Spread).")
    
    def calculate_ema(self, data_series, a=20):
        """
        Calcola la Media Mobile Esponenziale (EMA) per una serie di dati.
        
        Args:
            data_series (pd.Series): La serie di dati (prezzi o volumi).
            a (int): Il numero di periodi per l'EMA. Default 20.
        
        Returns:
            pd.Series: La serie di dati EMA.
        """
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
        moving_avg = data_series.rolling(window=b).mean()
        moving_std = data_series.rolling(window=b).std()
        z_score = (data_series - moving_avg) / moving_std
        return z_score

    def run_technical_analysis(self):
        """
        Esegue tutti i calcoli tecnici richiesti (su dati REALI) 
        e salva i risultati.
        """
        print("\nAvvio calcoli di analisi tecnica (su dati REALI)...")
        
        # 1. Calcolo EMA (sui VOLUMI REALI)
        print("  Calcolo EMA (a=20) per i VOLUMI di 'SPY', 'QQQ' (su dati REALI)...")
        self.results['ema'] = {}
        
        # I ticker base per cui cercare i volumi
        base_tickers = ['SPY', 'QQQ']
        
        for ticker in base_tickers:
            # Costruisci il nome della colonna volume come definito in MarketAnalyzer
            volume_col_name = f"{ticker}_Volume" 
            
            if volume_col_name in self.analyzer.data.columns:
                # Seleziona la serie dei volumi reali
                raw_volume = self.analyzer.data[volume_col_name]
                
                # Calcola l'EMA e salvalo
                result_key = f"{volume_col_name}_EMA_20"
                self.results['ema'][result_key] = self.calculate_ema(raw_volume, a=20)
                print(f"    -> EMA calcolata per {volume_col_name}")
            else:
                print(f"    Warning: Colonna {volume_col_name} non trovata in self.analyzer.data per EMA.")

        # 2. Calcolo Z-Score (su dati REALI)
        print("  Calcolo Z-Score (b=20) per 'DJ_SPREAD' (su dati REALI)...")
        self.results['zscore'] = {}
        
        if 'DJ_SPREAD' in self.analyzer.data.columns:
            spread_data = self.analyzer.data['DJ_SPREAD']
            self.results['zscore']['DJ_SPREAD_ZScore_20'] = self.calculate_zscore(spread_data, b=20)
            print("    -> Z-Score calcolato per DJ_SPREAD")
        else:
            print("    Warning: Colonna 'DJ_SPREAD' non trovata in self.analyzer.data per calcolo Z-Score.")
            
        print("\nAnalisi tecnica completata. I risultati sono in 'tech_analyzer.results'")


        
        