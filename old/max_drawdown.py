import pandas as pd
import numpy as np

# Definisci il nome del file (assicurati che sia nella stessa cartella)
file_name = "analisi_tecnica_train.xlsx"

# --- 1. Carica i Dati ---
try:
    df = pd.read_excel(file_name)

    # --- 2. Preparazione Dati ---
    # Verifica la presenza delle colonne necessarie
    if 'Date' not in df.columns or 'SPY' not in df.columns:
        print("Errore: Le colonne 'Date' o 'SPY' non sono state trovate nel file.")
    else:
        # Converti la colonna 'Date' in formato datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ordina per data (fondamentale per i calcoli time series)
        df = df.sort_values(by='Date')
        
        # Estrai i prezzi di 'SPY' e imposta la data come indice
        spy_prices = df.set_index('Date')['SPY']

        # Assicurati che i dati siano numerici e rimuovi eventuali NaN
        spy_prices = pd.to_numeric(spy_prices, errors='coerce').dropna()

        if spy_prices.empty:
            print("Errore: Nessun dato 'SPY' valido trovato dopo la pulizia.")
        else:
            # --- 3. Calcolo CAGR (Compound Annual Growth Rate) ---
            
            # Valori iniziali e finali
            start_price = spy_prices.iloc[0]
            end_price = spy_prices.iloc[-1]
            
            # Date iniziali e finali
            start_date = spy_prices.index[0]
            end_date = spy_prices.index[len(spy_prices) - 1]
            
            # Calcolo degli anni (usando 365.25 per tenere conto degli anni bisestili)
            num_years = (end_date - start_date).days / 365.25
            
            # Formula CAGR: (Valore Finale / Valore Iniziale) ^ (1 / Anni) - 1
            if num_years > 0:
                cagr = (end_price / start_price) ** (1 / num_years) - 1
            else:
                cagr = 0 
            
            # --- 4. Calcolo Max Drawdown (MDD) ---
            
            # 1. Calcola il massimo cumulativo (il picco corrente)
            cumulative_max = spy_prices.cummax()
            
            # 2. Calcola il drawdown (la percentuale di discesa dal picco)
            drawdown = (spy_prices - cumulative_max) / cumulative_max
            
            # 3. Trova il valore minimo (la discesa massima)
            max_drawdown = drawdown.min()

            # --- 5. Stampa Risultati ---
            print("\n--- Risultati Analisi SPY ---")
            print(f"Periodo Analizzato (SPY): {start_date.date()} a {end_date.date()}")
            print(f"Anni totali: {num_years:.2f}")
            print(f"Valore Iniziale SPY: {start_price:.2f}")
            print(f"Valore Finale SPY: {end_price:.2f}")
            print(f"CAGR (Tasso di crescita annuale composto): {cagr:.2%}")
            print(f"Max Drawdown (MDD): {max_drawdown:.2%}")

except FileNotFoundError:
    print(f"Errore: File non trovato. Assicurati che '{file_name}' sia nella directory corretta.")
except Exception as e:
    print(f"Si è verificato un errore: {e}")