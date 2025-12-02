# ddm_for_investment

## Descrizione
Questo progetto implementa una pipeline completa di trading algoritmico per l'analisi e l'investimento sugli ETF **SPY** e **QQQ**. Il sistema combina tecniche di feature engineering con algoritmi genetici per ottimizzare i parametri della strategia e valuta le performance tramite backtesting su dati storici.

## Caratteristiche Principali

* **Analisi Multi-Ticker:** Supporto nativo per l'analisi di SPY e QQQ, utilizzando indici di mercato ausiliari (come ^DJI e ^DJT) per arricchire il dataset.
* **Ottimizzazione Genetica:** Integrazione di un `GeneticOptimizer` per individuare automaticamente i migliori parametri operativi (popolazione: 40, generazioni: 40).
* **Backtesting Rigoroso:** Divisione automatica dei dati in set di *Training* (dal 2007-04-01) e *Test* (dal 2022-01-01) per validare la robustezza della strategia fuori campione.
* **Reporting Completo:** Generazione automatica di metriche di performance in formato CSV e visualizzazione grafica delle curve di equità (Equity Curves).

## Struttura del Progetto

Il codice sorgente si trova nella cartella `src/`:

* `main.py`: Script principale che orchestra l'intera pipeline (caricamento, ottimizzazione, backtest e reportistica).
* `data_loader.py`: Modulo per il recupero e la preparazione dei dati di mercato.
* `feature_engineer.py`: Modulo per la creazione di indicatori tecnici e feature per il modello.
* `genetic_optimizer.py`: Implementazione dell'algoritmo genetico per l'ottimizzazione dei parametri.
* `strategy_engine.py`: Motore che genera segnali di trading ed esegue la simulazione.
* `performance_analyzer.py`: Calcolo delle metriche finanziarie e statistiche.
* `plot.py`: Funzioni per la generazione e il salvataggio dei grafici comparativi.

## Installazione

Assicurati di avere Python installato, quindi installa le dipendenze necessarie tramite pip:

```bash
pip install -r src/requirements.txt
