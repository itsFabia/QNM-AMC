import pandas as pd
from pathlib import Path
import re

# ---- Helpers ---------------------------------------------------------------

def read_long_table(path: str):
    """
    Liest eine Long-Tabelle im Format Ticker;Field;Date;Value robust ein.
    - Handhabt ; als Trennzeichen
    - Repariert Dateien, die fälschlich als 1 Spalte eingelesen werden
    - Normalisiert Spaltennamen auf ['Ticker','Field','Date','Value']
    """
    # 1) Versuch: normal mit sep=';'
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8-sig', dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=';', dtype=str)

    # Falls nur eine Spalte existiert, ist das File „verklebt“
    if df.shape[1] == 1:
        col = df.columns[0]
        # Header-Zeile könnte "Ticker;Field;Date;Value" sein
        # -> nochmal manuell splitten
        df = pd.read_csv(path, header=None, dtype=str)
        df = df[0].str.split(';', expand=True)
        # Header setzen, falls vorhanden
        # Erster Datensatz ist meist Datenzeile, deshalb setzen wir direkte Namen
        df.columns = ['Ticker','Field','Date','Value']
    else:
        # Spalten normalisieren
        df.columns = [c.strip().title() for c in df.columns]
        # Erwartete Namen sicherstellen
        rename_map = {
            'Ticker':'Ticker', 'Field':'Field', 'Date':'Date', 'Value':'Value',
            'Wert':'Value'
        }
        df = df.rename(columns=rename_map)
        if set(['Ticker','Field','Date','Value']) - set(df.columns):
            # fallback: versuchen, erste vier Spalten zuzuordnen
            df = df.iloc[:, :4]
            df.columns = ['Ticker','Field','Date','Value']

    # Typen & Trim
    for c in ['Ticker','Field']:
        df[c] = df[c].astype(str).str.strip()

    # Datum parsen (dayfirst)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date'])

    # Value zu float
    def to_float(s):
        s = str(s).strip()
        if s == '' or s.lower().startswith('#n/a'):
            return None
        s = s.replace(' ', '').replace(',', '')  # tausender-Trenner weg
        try:
            return float(s)
        except:
            return None

    df['Value'] = df['Value'].map(to_float)
    df = df.dropna(subset=['Value'])

    return df

def tidy_feature_name(ticker: str, field: str) -> str:
    # Kürze Ticker (z.B. "ABBN SE Equity" -> "ABBN")
    base = re.sub(r'\s+SE\s+Equity$', '', ticker).strip()
    base = re.sub(r'\s+Index$', '', base).strip()
    base = re.sub(r'\s+Curncy$', '', base).strip()
    base = base.replace(' ', '')
    # Feldname aufräumen
    f = field.replace(' ', '').replace('-', '').replace('/', '')
    return f"{base}_{f}"

# ---- Pfade (anpassen, falls nötig) ----------------------------------------

stocks_long_path = r"data/stocks_long.csv"  # oder voller Pfad
macro_long_path  = r"data/macro_long.csv"

# ---- Einlesen (Semikolon-fest) --------------------------------------------

stocks_long = read_long_table(stocks_long_path)
macro_long  = read_long_table(macro_long_path)

# ---- Relevante Aktien-Felder auswählen (du kannst die Liste erweitern) ----

relevant_fields = [
    "Last Price", "Bid Price", "High Price", "Low Price",
    "Volume", "Current Market Cap",
    "BEst EBITDA", "BEst EPS", "BEst P/E Ratio", "Dividend 12 Month Yld - Gross",
]
stocks_f = stocks_long[stocks_long['Field'].isin(relevant_fields)].copy()

# Feature-Namen bauen
stocks_f['FeatureName'] = stocks_f.apply(
    lambda r: tidy_feature_name(r['Ticker'], r['Field']), axis=1
)

# Pivot: Datum als Index, jede Aktie*Feld als Spalte
stocks_wide = stocks_f.pivot_table(
    index='Date', columns='FeatureName', values='Value', aggfunc='first'
)

# ---- Makro pivotieren ------------------------------------------------------

# Makronamen: Ticker als Variablen-Name, leicht gesäubert
macro_long['FeatureName'] = macro_long['Ticker'].str.replace(' Index','', regex=False)\
                                               .str.replace(' Curncy','', regex=False)\
                                               .str.replace(' ','', regex=False)
macro_wide = macro_long.pivot_table(
    index='Date', columns='FeatureName', values='Value', aggfunc='first'
)

# ---- Merge: Aktien + Makro pro Datum --------------------------------------

combined = pd.merge(stocks_wide, macro_wide, on='Date', how='inner')
combined = combined.sort_index()

# Ausgabe
out_path = Path("qnm_step2_combined.csv")
combined.to_csv(out_path)

print("Erstellt:", out_path)
print("Shape:", combined.shape)
print("Beispielspalten:", list(combined.columns)[:15])
