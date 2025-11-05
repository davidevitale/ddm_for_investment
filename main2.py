from data2 import MarketAnalyzer
from datetime import datetime
from indicators2 import TechnicalAnalyzer
import pandas as pd


# --- Esempio di utilizzo ---

tickers_to_analyze = {
    '^DJI': 'Dow Jones Industrial Average',
    '^DJT': 'Dow Jones Transportation Average',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust (Nasdaq-100)'
}

start_date = '2007-04-10'
end_date = datetime.now().strftime('%Y-%m-%d')

# Crea l'istanza e avvia l'analisi
analyzer = MarketAnalyzer(tickers_to_analyze, start_date, end_date)
analyzer.run_analysis()

# 3. Esecuzione del TechnicalAnalyzer (passando l'istanza 'analyzer')
tech_analyzer = TechnicalAnalyzer(analyzer)
tech_analyzer.run_technical_analysis()
dfs_to_combine = [analyzer.data]

# 4. Controllo dei risultati
print("\n--- Risultati Tecnici (Anteprima) ---")

if 'ema' in tech_analyzer.results and tech_analyzer.results['ema']:
    print("EMA (ultime 5 righe):")
    # Combina le EMA in un DataFrame per una visualizzazione pulita
    ema_df = pd.DataFrame(tech_analyzer.results['ema'])
    print(ema_df.tail())
    dfs_to_combine.append(ema_df)
    
if 'zscore' in tech_analyzer.results and tech_analyzer.results['zscore']:
    print("\nZ-Score (ultime 5 righe):")
    # Combina gli Z-Score in un DataFrame
    zscore_df = pd.DataFrame(tech_analyzer.results['zscore'])
    print(zscore_df.head())
    dfs_to_combine.append(zscore_df)

# 4. Concatena tutto lungo l'asse delle colonne (axis=1)
# Questo funziona perché tutti i DataFrame condividono lo stesso indice (Date)
dataset = pd.concat(dfs_to_combine, axis=1)

# 5. Mostra il risultato
print("\n--- Anteprima DataFrame Completo (prime 5 righe) ---")
print(dataset.head())
