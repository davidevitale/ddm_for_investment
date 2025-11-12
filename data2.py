# Import required libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")
print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}")


class MarketAnalyzer:
    """
    Una classe per scaricare, analizzare e plottare i dati di mercato da yfinance.
    Tutti i dati (prezzi e volumi) sono memorizzati in self.data.
    L'analisi viene eseguita su un unico dataset completo (nessun split train/test).
    """
    
    def __init__(self, tickers_dict, start_date, end_date):
        """
        Inizializza l'analizzatore.
        """
        self.tickers_dict = tickers_dict
        self.tickers = list(tickers_dict.keys()) # Lista dei ticker originali (solo prezzi)
        self.start_date = start_date
        self.end_date = end_date
        self.data = None # Conterrà TUTTE le features (prezzi, volumi, calcolati)
        
        # Attributi per i dati normalizzati (Base 100)
        self.normalized_prices = None
        self.normalized_volumes = None
        self.normalized_data = None # DataFrame normalizzato COMBINATO
        
        print(f"MarketAnalyzer initialized for {len(self.tickers)} assets.")
        print("Investment Universe:")
        for ticker, description in self.tickers_dict.items():
            print(f"  {ticker}: {description}")
        print(f"\nAnalysis Period: {self.start_date} to {self.end_date}")

    def download_data(self):
        """
        Scarica i dati dei prezzi e volumi selezionati da yfinance.
        Usa 'Adj Close' per i prezzi.
        Salva 'Volume' per 'SPY' e 'QQQ' rinominandolo (es. 'SPY_Volume').
        Combina tutto in self.data.
        """
        print("\nDownloading price and volume data...")
        all_frames = {} # Unico dizionario per tutte le features
        
        for ticker in self.tickers:
            try:
                # Scarica i dati completi
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if data.empty:
                    print(f"  Warning: No data found for {ticker}.")
                    continue
                
                # --- Gestione Prezzi ---
                if 'Adj Close' in data.columns:
                    all_frames[ticker] = data['Adj Close']
                else:
                    all_frames[ticker] = data['Close']
                
                print(f"  {ticker}: {len(data)} observations from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

                # --- Gestione Volumi (con rinomina) ---
                if ticker in ['SPY', 'QQQ'] and 'Volume' in data.columns:
                    col_name = f"{ticker}_Volume"
                    all_frames[col_name] = data['Volume']
                    print(f"    -> Volume data captured as '{col_name}'")

            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")

        if not all_frames:
            print("No data downloaded. Exiting.")
            return

        # Combina tutte le serie (prezzi e volumi) in un unico DataFrame
        self.data = pd.concat(all_frames.values(), axis=1)
        self.data.columns = all_frames.keys() # Assicura che i nomi colonna siano corretti

        # Trova la data di inizio comune più recente per tutti gli asset
        price_frames_list = [all_frames[t] for t in self.tickers if t in all_frames]
        if not price_frames_list:
            print("No valid price data downloaded. Exiting.")
            return
            
        latest_start_date = max(df.index[0] for df in price_frames_list)
        print(f"\nCommon start date (when all assets are available): {latest_start_date.strftime('%Y-%m-%d')}")

        # Filtra i dati dalla data di inizio comune e rimuovi eventuali righe con NaN
        self.data = self.data.loc[latest_start_date:]
        self.data = self.data.dropna() 

        print(f"\nCombined features dataset: {len(self.data)} observations")
        print(f"Final date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")

    def calculate_spread_djia_djt(self):
        """
        Calcola lo spread DJIA - DJT e lo aggiunge a self.data.
        """
        if self.data is None:
            print("Dati non ancora scaricati. Impossibile calcolare lo spread.")
            return
        
        required_tickers = ['^DJI', '^DJT']
        if all(ticker in self.data.columns for ticker in required_tickers):
            self.data['DJ_SPREAD'] = self.data['^DJI'] - self.data['^DJT']
        else:
            print("Warning: Ticker ^DJI or ^DJT non trovati. Impossibile calcolare 'DJ_SPREAD'.")

    def get_statistics(self):
        """
        Restituisce un DataFrame con le statistiche descrittive di self.data (l'intero set).
        """
        if self.data is None:
            print("Data not found.")
            return None
        return self.data.describe().round(2)

    def calculate_normalized_prices(self, base=100):
        """
        Normalizza i PREZZI (self.data) a Base 100,
        basandosi sulla prima riga del dataset.
        """
        print("\nCalculating price normalization (Base 100) based on full dataset...")
        if self.data is None or self.data.empty:
            print("Data not found or is empty. Cannot normalize prices.")
            return
        
        # Seleziona solo le colonne dei prezzi originali
        original_ticker_prices = self.data[self.tickers]
        base_values = original_ticker_prices.iloc[0]
        
        self.normalized_prices = (original_ticker_prices / base_values) * base
        print("Price data normalized.")

    def calculate_normalized_volumes(self, base=100):
        """
        Normalizza i VOLUMI (self.data) a Base 100,
        basandosi sulla prima riga del dataset.
        """
        print("\nCalculating volume normalization (Base 100) based on full dataset...")
        if self.data is None or self.data.empty:
            print("Data not found or is empty. Cannot normalize volumes.")
            return

        volume_cols = [col for col in self.data.columns if col.endswith('_Volume')]
        if not volume_cols:
            print("No volume columns found to normalize.")
            return

        original_volumes = self.data[volume_cols]
        base_values = original_volumes.iloc[0]
        # Evita divisione per zero se il primo volume fosse 0
        base_values[base_values == 0] = 1 
        
        self.normalized_volumes = (original_volumes / base_values) * base
        print("Volume data normalized (Base 100).")

    def combine_normalized_data(self):
        """
        Combina prezzi normalizzati e volumi normalizzati in un unico DataFrame.
        """
        print("\nCombining normalized price and volume data...")
        
        if self.normalized_prices is not None and self.normalized_volumes is not None:
            self.normalized_data = pd.concat([self.normalized_prices, self.normalized_volumes], axis=1)
            print("Normalized data combined (prices and volumes).")
        elif self.normalized_prices is not None:
            self.normalized_data = self.normalized_prices
            print("Normalized data combined (only prices).")
        else:
            print("No normalized data to combine.")

    def plot_normalized_prices(self):
        """
        Plotta l'evoluzione dei prezzi normalizzati (dataset completo).
        """
        if self.normalized_prices is None or self.normalized_prices.empty:
            print("Normalized prices not available. Cannot plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 9))
        
        for ticker in self.normalized_prices.columns:
            label = f"{ticker} - {self.tickers_dict.get(ticker, ticker)}"
            ax.plot(self.normalized_prices.index, self.normalized_prices[ticker], 
                    label=label, linewidth=2)
        
        ax.set_title('Normalized Price Evolution (Base = 100 on Start Date)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        """
        Esegue l'intero flusso di analisi sul dataset completo.
        """
        self.download_data()
        
        if self.data is not None and not self.data.empty:
            
            # 1. Calcola le feature sull'intero set
            self.calculate_spread_djia_djt()
            
            # 2. Stampa statistiche (sull'intero set)
            print("\n--- Descriptive Statistics (Full Dataset) ---")
            stats = self.get_statistics()
            if stats is not None:
                print(stats)
            
            # Stampa l'anteprima del set RAW
            print("\n--- Raw Data (Full Set Head) ---")
            print(self.data.head())
            
            # 3. Normalizza (basandoti sull'intero set)
            self.calculate_normalized_prices()
            self.calculate_normalized_volumes() 
            
            # 4. Combina i dati normalizzati
            self.combine_normalized_data()
            
            # 5. Plotta
            self.plot_normalized_prices()
        else:
            print("Analysis failed: No data could be processed.")


