"""
Yahoo Finance Fundamentals Downloader für SMI-Aktien
Lädt Fundamentals (P/E, P/B, ROE, etc.) und merged sie mit existierenden Daten
"""
import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TICKER MAPPING: Bloomberg -> Yahoo Finance
# ============================================================

TICKER_MAPPING = {
    # Bloomberg Ticker: Yahoo Finance Symbol
    'ABBN SE Equity': 'ABBN.SW',      # ABB
    'ALC SE Equity': 'ADEN.SW',       # Adecco (ALC = Adecco ticker code)
    'AMRZ SE Equity': None,           # Partners Group? (need to check)
    'CFR SE Equity': 'CFR.SW',        # Richemont
    'GEBN SE Equity': 'GEBN.SW',      # Geberit
    'GIVN SE Equity': 'GIVN.SW',      # Givaudan
    'HOLN SE Equity': 'HOLN.SW',      # Holcim
    'KNIN SE Equity': 'KNIN.SW',      # Kuehne+Nagel
    'LOGN SE Equity': 'LOGN.SW',      # Logitech
    'LONN SE Equity': 'LONN.SW',      # Lonza
    'NESN SE Equity': 'NESN.SW',      # Nestle
    'NOVN SE Equity': 'NOVN.SW',      # Novartis
    'PGHN SE Equity': None,           # Partners Group? (need to check)
    'ROG SE Equity': 'ROG.SW',        # Roche
    'SCMN SE Equity': 'SCMN.SW',      # Swisscom
    'SIKA SE Equity': 'SIKA.SW',      # Sika
    'SLHN SE Equity': 'SLHN.SW',      # Swiss Life
    'SREN SE Equity': 'SREN.SW',      # Swiss Re
    'UBSG SE Equity': 'UBSG.SW',      # UBS
    'ZURN SE Equity': 'ZURN.SW',      # Zurich Insurance
}

# ============================================================
# FUNDAMENTALS DOWNLOAD
# ============================================================

def get_quarterly_fundamentals(yf_symbol, max_periods=40):
    """
    Lädt quarterly fundamentals für ein Symbol
    Returns: DataFrame mit Date als Index
    """
    print(f"  [INFO] Lade {yf_symbol}...")

    try:
        ticker = yf.Ticker(yf_symbol)

        # Quarterly Financials
        quarterly_financials = ticker.quarterly_financials
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow

        if quarterly_financials is None or quarterly_financials.empty:
            print(f"  [WARN] Keine Daten für {yf_symbol}")
            return None

        # Transpose (Dates as rows)
        qf = quarterly_financials.T
        qb = quarterly_balance.T if quarterly_balance is not None else pd.DataFrame()
        qc = quarterly_cashflow.T if quarterly_cashflow is not None else pd.DataFrame()

        # Combine
        df = pd.concat([qf, qb, qc], axis=1)

        # Reset index (Date column)
        df = df.reset_index().rename(columns={'index': 'Date'})

        # Limit to recent periods
        df = df.head(max_periods)

        print(f"  [OK] {len(df)} Perioden geladen")
        return df

    except Exception as e:
        print(f"  [ERROR] {yf_symbol}: {e}")
        return None


def get_current_fundamentals(yf_symbol):
    """
    Lädt aktuelle Fundamentals (Ratios, etc.)
    Returns: Dict
    """
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        return {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'peg_ratio': info.get('pegRatio'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'ev_revenue': info.get('enterpriseToRevenue'),

            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'roic': None,  # Not directly available

            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'gross_margin': info.get('grossMargins'),

            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),

            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),

            'earnings_growth': info.get('earningsQuarterlyGrowth'),
            'revenue_growth': info.get('revenueGrowth'),

            'beta': info.get('beta'),
            'market_cap': info.get('marketCap'),

            'book_value_per_share': info.get('bookValue'),
            'earnings_per_share': info.get('trailingEps'),
        }

    except Exception as e:
        print(f"  [ERROR] Current fundamentals for {yf_symbol}: {e}")
        return {}


def download_all_fundamentals(ticker_mapping, output_dir='data/fundamentals'):
    """
    Lädt Fundamentals für alle Ticker
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Lade Fundamentals für {len(ticker_mapping)} Stocks...")

    all_data = {}

    for bb_ticker, yf_symbol in ticker_mapping.items():
        if yf_symbol is None:
            print(f"  [SKIP] {bb_ticker} (kein Yahoo-Symbol)")
            continue

        # Quarterly Fundamentals
        quarterly_df = get_quarterly_fundamentals(yf_symbol)

        # Current Fundamentals
        current = get_current_fundamentals(yf_symbol)

        if quarterly_df is not None or current:
            all_data[bb_ticker] = {
                'yf_symbol': yf_symbol,
                'quarterly': quarterly_df,
                'current': current
            }

        # Rate limiting (Yahoo Finance has limits)
        time.sleep(0.5)

    print(f"[INFO] Erfolgreich geladen: {len(all_data)}/{len(ticker_mapping)} Stocks")

    # Save raw data
    import pickle
    with open(output_dir / 'fundamentals_raw.pkl', 'wb') as f:
        pickle.dump(all_data, f)

    print(f"[INFO] Gespeichert: {output_dir / 'fundamentals_raw.pkl'}")

    return all_data


# ============================================================
# MERGE MIT EXISTIERENDEN DATEN
# ============================================================

def create_fundamental_features(all_fundamentals_data):
    """
    Erstellt DataFrame mit Fundamental-Features pro Ticker und Datum
    """
    print("[INFO] Erstelle Fundamental-Features...")

    rows = []

    for bb_ticker, data in all_fundamentals_data.items():
        current = data['current']

        # Für jedes verfügbare Feature
        feature_row = {
            'Ticker': bb_ticker,
            'Date': pd.Timestamp.now(),  # Current snapshot

            # Valuation Ratios
            'PE_Ratio': current.get('pe_ratio'),
            'Forward_PE': current.get('forward_pe'),
            'PB_Ratio': current.get('pb_ratio'),
            'PS_Ratio': current.get('ps_ratio'),
            'PEG_Ratio': current.get('peg_ratio'),
            'EV_EBITDA': current.get('ev_ebitda'),
            'EV_Revenue': current.get('ev_revenue'),

            # Profitability
            'ROE': current.get('roe'),
            'ROA': current.get('roa'),
            'Profit_Margin': current.get('profit_margin'),
            'Operating_Margin': current.get('operating_margin'),
            'Gross_Margin': current.get('gross_margin'),

            # Financial Health
            'Debt_to_Equity': current.get('debt_to_equity'),
            'Current_Ratio': current.get('current_ratio'),
            'Quick_Ratio': current.get('quick_ratio'),

            # Dividend
            'Dividend_Yield_YF': current.get('dividend_yield'),
            'Payout_Ratio': current.get('payout_ratio'),

            # Growth
            'Earnings_Growth_QoQ': current.get('earnings_growth'),
            'Revenue_Growth': current.get('revenue_growth'),

            # Risk
            'Beta': current.get('beta'),

            # Per-Share
            'Book_Value_Per_Share': current.get('book_value_per_share'),
            'EPS': current.get('earnings_per_share'),
        }

        rows.append(feature_row)

    df_fundamentals = pd.DataFrame(rows)

    print(f"[INFO] Features erstellt: {len(df_fundamentals)} Stocks, {df_fundamentals.shape[1]-2} Features")

    return df_fundamentals


def merge_with_existing_data(input_csv, fundamentals_df, output_csv):
    """
    Merged Fundamentals mit existierenden Price/Volume Daten
    """
    print(f"[INFO] Lade existierende Daten: {input_csv}")

    try:
        df = pd.read_csv(input_csv, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(input_csv)
    except:
        df = pd.read_csv(input_csv)

    print(f"[INFO] Existierende Daten: {df.shape}")

    # Backup
    backup_path = Path(output_csv).parent / (Path(output_csv).stem + "_BEFORE_FUNDAMENTALS.csv")
    df.to_csv(backup_path, index=False)
    print(f"[INFO] Backup erstellt: {backup_path}")

    # Merge (broadcast fundamentals to all dates)
    # Da wir nur current snapshot haben, fügen wir diese zu allen Zeilen des Tickers hinzu

    print("[INFO] Merge Fundamentals mit Price Data...")

    # Prepare fundamentals for merge (drop Date column)
    fundamentals_for_merge = fundamentals_df.drop(columns=['Date'])

    # Merge on Ticker
    df_merged = df.merge(fundamentals_for_merge, on='Ticker', how='left')

    print(f"[INFO] Nach Merge: {df_merged.shape}")
    print(f"[INFO] Neue Features: {df_merged.shape[1] - df.shape[1]}")

    # Save
    df_merged.to_csv(output_csv, index=False)
    print(f"[INFO] Gespeichert: {output_csv}")

    return df_merged


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("YAHOO FINANCE FUNDAMENTALS DOWNLOADER")
    print("="*70)

    # Step 1: Download Fundamentals
    all_fundamentals = download_all_fundamentals(TICKER_MAPPING)

    # Step 2: Create Feature DataFrame
    fundamentals_df = create_fundamental_features(all_fundamentals)

    # Save intermediate
    fundamentals_df.to_csv('data/fundamentals/fundamentals_snapshot.csv', index=False)
    print(f"[INFO] Snapshot gespeichert: data/fundamentals/fundamentals_snapshot.csv")

    # Step 3: Merge with existing data
    input_file = 'data/AMC_model_input_BEFORE_COMPREHENSIVE.csv'  # Use version before 195 features
    output_file = 'data/AMC_model_input_with_fundamentals.csv'

    df_merged = merge_with_existing_data(input_file, fundamentals_df, output_file)

    # Summary
    print("\n" + "="*70)
    print("ZUSAMMENFASSUNG")
    print("="*70)
    print(f"Stocks mit Fundamentals: {len(all_fundamentals)}/20")
    print(f"Fundamental Features: {fundamentals_df.shape[1]-2}")
    print(f"Final Dataset: {df_merged.shape}")
    print(f"\nFundamental Features:")
    for col in fundamentals_df.columns[2:]:  # Skip Ticker, Date
        print(f"  - {col}")

    print("\n[DONE] Fundamentals erfolgreich heruntergeladen und gemerged!")
    print(f"\nNächster Schritt:")
    print(f"  python scripts/clean_and_reduce.py --input data/AMC_model_input_with_fundamentals.csv \\")
    print(f"    --output data/AMC_model_input_reduced.csv --min-samples 500")


if __name__ == "__main__":
    main()
