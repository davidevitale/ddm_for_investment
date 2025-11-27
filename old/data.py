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
    """
    
    def __init__(self, tickers_dict, start_date, end_date):
        """
        Inizializza l'analizzatore.
        
        Args:
            tickers_dict (dict): Dizionario {ticker: descrizione}
            start_date (str): Data di inizio (YYYY-MM-DD)
            end_date (str): Data di fine (YYYY-MM-DD)
        """
        self.tickers_dict = tickers_dict
        self.tickers = list(tickers_dict.keys()) # Lista dei ticker originali
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None # Conterrà i prezzi E le nuove colonne
        self.normalized_prices = None # Conterrà SOLO i prezzi normalizzati
        
        print(f"MarketAnalyzer initialized for {len(self.tickers)} assets.")
        print("Investment Universe:")
        for ticker, description in self.tickers_dict.items():
            print(f"  {ticker}: {description}")
        print(f"\nAnalysis Period: {self.start_date} to {self.end_date}")

    def download_data(self):
        """
        Scarica i dati dei prezzi da yfinance.
        Usa 'Adj Close' se disponibile, altrimenti 'Close'.
        """
        print("\nDownloading price data...")
        price_frames = {}
        
        for ticker in self.tickers:
            try:
                # Scarica i dati completi
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if data.empty:
                    print(f"  Warning: No data found for {ticker}.")
                    continue
                
                # Seleziona 'Adj Close' se esiste, altrimenti 'Close'
                if 'Adj Close' in data.columns:
                    price_frames[ticker] = data['Adj Close']
                else:
                    price_frames[ticker] = data['Close']
                
                print(f"  {ticker}: {len(data)} observations from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")

        if not price_frames:
            print("No data downloaded. Exiting.")
            return

        # Combina tutte le serie in un unico DataFrame
        self.price_data = pd.concat(price_frames.values(), axis=1)
        self.price_data.columns = price_frames.keys() # Assicura che i nomi colonna siano i ticker

        # Trova la data di inizio comune più recente per tutti gli asset
        # Nota: usiamo .values() per evitare problemi se un ticker non ha scaricato dati
        valid_frames = [df for df in price_frames.values() if not df.empty]
        if not valid_frames:
             print("No valid data downloaded. Exiting.")
             return

        latest_start_date = max(df.index[0] for df in valid_frames)
        print(f"\nCommon start date (when all assets are available): {latest_start_date.strftime('%Y-%m-%d')}")

        # Filtra i dati dalla data di inizio comune e rimuovi eventuali righe con NaN
        self.price_data = self.price_data.loc[latest_start_date:]
        self.price_data = self.price_data.dropna() 

        print(f"\nCombined dataset: {len(self.price_data)} observations")
        print(f"Final date range: {self.price_data.index[0].strftime('%Y-%m-%d')} to {self.price_data.index[-1].strftime('%Y-%m-%d')}")

    def calculate_spread_djia_djt(self):
        """
        Calcola lo spread DJIA - DJT e lo aggiunge a price_data.
        """
        if self.price_data is None:
            print("Dati non ancora scaricati. Impossibile calcolare lo spread.")
            return
        
        required_tickers = ['^DJI', '^DJT']
        if all(ticker in self.price_data.columns for ticker in required_tickers):
            self.price_data['DJ_SPREAD'] = self.price_data['^DJI'] - self.price_data['^DJT']
        else:
            print("Warning: Ticker ^DJI or ^DJT non trovati. Impossibile calcolare 'DJ_SPREAD'.")

    def calculate_ratio_vvix_vix(self):
        """
        Calcola il ratio VVIX / VIX e lo aggiunge a price_data.
        """
        if self.price_data is None:
            print("Dati non ancora scaricati. Impossibile calcolare il ratio.")
            return
        required_tickers = ['^VVIX', '^VIX']
        if all(ticker in self.price_data.columns for ticker in required_tickers):
            self.price_data['VVIX_VIX_RATIO'] = self.price_data['^VVIX'] / self.price_data['^VIX']
        else:
            print("Warning: Ticker ^VVIX or ^VIX non trovati. Impossibile calcolare 'VVIX_VIX_RATIO'.")

    def calculate_normalized_prices(self, base=100):
        """
        Normalizza i prezzi a una base comune (es. 100).
        """
        if self.price_data is None or self.price_data.empty:
            print("Price data not found or is empty. Cannot normalize.")
            return
        
        # Seleziona solo i ticker per la normalizzazione
        original_ticker_prices = self.price_data[self.tickers]
        self.normalized_prices = original_ticker_prices / original_ticker_prices.iloc[0] * base

    def get_statistics(self):
        """
        Restituisce un DataFrame con le statistiche descrittive.
        """
        if self.price_data is None:
            print("Price data not found.")
            return None
        return self.price_data.describe().round(2)

    def plot_normalized_prices(self):
        """
        Plotta l'evoluzione dei prezzi normalizzati.
        """
        if self.normalized_prices is None or self.normalized_prices.empty:
            print("Normalized prices not available. Cannot plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 9))
        # Itera solo sulle colonne dei prezzi normalizzati
        for ticker in self.normalized_prices.columns:
            label = f"{ticker} - {self.tickers_dict.get(ticker, ticker)}"
            ax.plot(self.normalized_prices.index, self.normalized_prices[ticker], 
                    label=label, linewidth=2)
            
        ax.set_title('Normalized Price Evolution (Base = 100)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Price', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        """
        Esegue l'intero flusso di analisi.
        """
        self.download_data()
        
        if self.price_data is not None and not self.price_data.empty:
            
            self.calculate_spread_djia_djt()
            self.calculate_ratio_vvix_vix()
            stats = self.get_statistics()
            if stats is not None:
                print(stats)

            self.calculate_normalized_prices()
            self.plot_normalized_prices()
        else:
            print("Analysis failed: No data could be processed.")

# --- ESECUZIONE DELLO SCRIPT ---

# 1. Definisci l'universo di investimento
# (Ticker YFinance: Descrizione)
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
# (VVIX è disponibile circa dal 2007)
start_date = '2007-04-10'
end_date = datetime.now().strftime('%Y-%m-%d')

# 3. Crea l'istanza e avvia l'analisi
try:
    analyzer = MarketAnalyzer(tickers_dict=new_data, start_date=start_date, end_date=end_date)
    analyzer.run_analysis()
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

