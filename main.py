from datetime import datetime
from data import MarketAnalyzer
from technical_indicators import TechnicalAnalyzer

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