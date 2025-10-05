# -*- coding: utf-8 -*-
"""
SMI-CSV aufbereiten:
- Spaltennamen vereinheitlichen
- Wide -> Long
- Feature Engineering (Returns, SMAs, Volatilitaet, optional RSI/Bollinger/MACD)
- CSV speichern (Orange- und Excel-freundlich)

Voraussetzungen:
    pip install pandas numpy
Optional:
    pip install yfinance
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ========= Einstellungen =========
# Quelle: vorhandene CSV aus yfinance-Export (mit ; oder , getrennt)
SRC = Path(r"C:/Users/holze/PycharmProjects/QNM-AMC/smi_stocks_full.csv")
OUT_DIR = SRC.parent

# Feature-Parameter
SMA_SHORT = 20
SMA_LONG = 50
VOL_WIN = 20
ADD_RSI = True
RSI_WIN = 14
ADD_BBANDS = True
BB_WIN = 20
BB_K = 2.0
ADD_MACD = True
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9


# ========= Hilfsfunktionen =========
def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Liest eine CSV ein und erkennt das Trennzeichen automatisch.
    Parsed 'Date' robust mit dayfirst=True (CH/DE-Format).
    """
    # Erst versuchen mit auto-sep
    df = pd.read_csv(path, sep=None, engine="python")
    # Falls 'Date' existiert: als Datum parsen (dayfirst)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        if df["Date"].isna().all():
            # vielleicht ist die erste Spalte das Datum
            pass
    else:
        # Falls erste Spalte Datum ist, umbenennen
        first = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first], dayfirst=True, errors="coerce")
            if parsed.notna().any():
                df.rename(columns={first: "Date"}, inplace=True)
                df["Date"] = parsed
        except Exception:
            pass

    if "Date" not in df.columns:
        raise ValueError("Konnte keine Datumsspalte erkennen. Pruefe die CSV (Spalte 'Date').")

    # Nur Zeilen mit gueltigem Datum behalten
    df = df[df["Date"].notna()].copy()
    # Nach Datum sortieren
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet Spalten wie 'Open_ABBN.SW' oder 'Adj Close_NESN.SW'.
    Macht daraus 'Open_ABBN.SW' bzw. 'AdjClose_NESN.SW' (Leerzeichen raus).
    Kann auch rudimentaer mit Tuple-Strings umgehen.
    """
    new_cols = []
    for c in df.columns:
        if c == "Date":
            new_cols.append(c)
            continue

        # Bereits mit Unterstrich?
        if "_" in c:
            left, right = c.rsplit("_", 1)
            field = left.strip().replace(" ", "")
            new_cols.append(f"{field}_{right}")
            continue

        # Versuche Tuple-artige Spalten ("('Adj Close', 'NESN.SW')")
        cc = str(c).strip()
        if cc.startswith("(") and "," in cc:
            cc = cc.strip("()").replace("'", "").replace('"', "")
            parts = [p.strip() for p in cc.split(",")]
            if len(parts) >= 2:
                field = parts[0].replace(" ", "")
                ticker = parts[1]
                new_cols.append(f"{field}_{ticker}")
                continue

        # Fallback: so belassen
        new_cols.append(c)

    df.columns = new_cols
    return df


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=price.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=price.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bbands(price: pd.Series, win: int = 20, k: float = 2.0):
    mid = price.rolling(win, min_periods=1).mean()
    std = price.rolling(win, min_periods=1).std()
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width


def compute_macd(price: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist


# ========= Pipeline =========
# 1) Laden
df = read_csv_smart(SRC)

# 2) Spalten normalisieren
df = normalize_columns(df)

# 3) Saubere Wide-Versionen speichern
wide_orange = OUT_DIR / "smi_wide_orange.csv"   # , getrennt
wide_excel = OUT_DIR / "smi_wide_excel.csv"     # ; getrennt
df.to_csv(wide_orange, index=False)
df.to_csv(wide_excel, index=False, sep=";")

# 4) Wide -> Long (melt)
long_df = df.melt(id_vars=["Date"], var_name="Field_Ticker", value_name="Value")
split = long_df["Field_Ticker"].str.rsplit("_", n=1, expand=True)
long_df["Field"] = split[0]
long_df["Ticker"] = split[1]
long_df = long_df.drop(columns=["Field_Ticker"])
long_df = long_df[["Date", "Ticker", "Field", "Value"]].sort_values(["Ticker", "Date", "Field"])

long_path = OUT_DIR / "smi_long.csv"
long_df.to_csv(long_path, index=False)

# 5) Feature Engineering auf Basis von AdjClose (Fallback auf Close) + Volume
# Preis-Matrix (Wide) aufbauen
price_cols = [c for c in df.columns if c.startswith("AdjClose_")]
if not price_cols:
    price_cols = [c for c in df.columns if c.startswith("Close_")]
if not price_cols:
    raise ValueError("Keine Preis-Spalten gefunden (AdjClose_* oder Close_*).")

price_w = df[["Date"] + price_cols].copy().sort_values("Date")
price_w.set_index("Date", inplace=True)
# Spalten: nur Ticker
price_w.columns = [c.split("_", 1)[1] for c in price_w.columns]

# Volumen (optional)
vol_cols = [c for c in df.columns if c.startswith("Volume_")]
volume_w = None
if vol_cols:
    volume_w = df[["Date"] + vol_cols].copy().sort_values("Date").set_index("Date")
    volume_w.columns = [c.split("_", 1)[1] for c in volume_w.columns]

# 6) Pro Ticker Features rechnen (Long-Struktur)
feat_frames = []
for tkr in price_w.columns:
    s = price_w[tkr]

    f = pd.DataFrame(index=s.index)
    f["Ticker"] = tkr
    f["Price"] = s
    f["Ret"] = s.pct_change()
    f["LogRet"] = np.log(s / s.shift(1))

    # SMAs
    f[f"SMA{SMA_SHORT}"] = s.rolling(SMA_SHORT, min_periods=1).mean()
    f[f"SMA{SMA_LONG}"] = s.rolling(SMA_LONG, min_periods=1).mean()

    # Rollierende Volatilitaet auf LogRet
    f[f"Vol{VOL_WIN}"] = f["LogRet"].rolling(VOL_WIN, min_periods=1).std()

    # RSI
    if ADD_RSI:
        f[f"RSI{RSI_WIN}"] = compute_rsi(s, window=RSI_WIN)

    # Bollinger
    if ADD_BBANDS:
        mid, up, lo, width = compute_bbands(s, win=BB_WIN, k=BB_K)
        f[f"BBmid{BB_WIN}"] = mid
        f[f"BBup{BB_WIN}"] = up
        f[f"BBlo{BB_WIN}"] = lo
        f[f"BBwidth{BB_WIN}"] = width

    # MACD
    if ADD_MACD:
        macd, macd_sig, macd_hist = compute_macd(s, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
        f["MACD"] = macd
        f["MACDsig"] = macd_sig
        f["MACDhist"] = macd_hist

    # Volume-Features
    if volume_w is not None and tkr in volume_w.columns:
        f["Volume"] = volume_w[tkr]
        f[f"VolAvg{SMA_SHORT}"] = volume_w[tkr].rolling(SMA_SHORT, min_periods=1).mean()

    f = f.reset_index().rename(columns={"index": "Date"})
    feat_frames.append(f)

features_long = pd.concat(feat_frames, ignore_index=True)

# sinnvolle Spaltenreihenfolge
base_cols = ["Date", "Ticker", "Price", "Ret", "LogRet",
             f"SMA{SMA_SHORT}", f"SMA{SMA_LONG}", f"Vol{VOL_WIN}"]
opt_cols = []
if ADD_RSI: opt_cols += [f"RSI{RSI_WIN}"]
if ADD_BBANDS: opt_cols += [f"BBmid{BB_WIN}", f"BBup{BB_WIN}", f"BBlo{BB_WIN}", f"BBwidth{BB_WIN}"]
if ADD_MACD: opt_cols += ["MACD", "MACDsig", "MACDhist"]
if "Volume" in features_long.columns: opt_cols += ["Volume", f"VolAvg{SMA_SHORT}"]

features_long = features_long[base_cols + opt_cols]

# Speichern: Long-Features
features_long_path = OUT_DIR / "smi_features_long.csv"
features_long.to_csv(features_long_path, index=False)

# 7) Wide-Features fuer Orange (jede Feature/Ticker-Kombi eigene Spalte)
wide_feat = features_long.pivot_table(index="Date",
                                      columns="Ticker",
                                      values=features_long.columns.difference(["Date", "Ticker"]))
# ('Ret','NESN.SW') -> 'NESN.SW_Ret'
wide_feat.columns = [f"{tkr}_{metric}" for metric, tkr in wide_feat.columns]
wide_feat = wide_feat.sort_index()

features_wide_path = OUT_DIR / "smi_features_wide.csv"
features_wide_path_excel = OUT_DIR / "smi_features_wide_excel.csv"
wide_feat.to_csv(features_wide_path)              # , fuer Orange
wide_feat.to_csv(features_wide_path_excel, sep=";")  # ; fuer Excel

# 8) Kurzer Abschluss-Print
print("Gespeichert:")
print(f"- Wide roh (Orange): {wide_orange}")
print(f"- Wide roh (Excel):  {wide_excel}")
print(f"- Long roh:          {long_path}")
print(f"- Features long:     {features_long_path}")
print(f"- Features wide:     {features_wide_path}")
print(f"- Features wide (;): {features_wide_path_excel}")
