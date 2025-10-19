import pandas as pd
import numpy as np
from pathlib import Path
import re

# -----------------------------------------
# Pfade & Parameter
# -----------------------------------------
STOCKS_PATH = Path("../data/stocks_long.csv")     # Ticker, Field, Date, Value
MACRO_PATH  = Path("../data/macro_long.csv")      # Field, Date, Value
SMI_PATH    = Path("../data/SMI data.csv")        # optional für SMI_Return
OUT_PATH    = Path("qnm_step5_SMI_5d_modelinput.csv")

ROLL_WINDOWS   = [5, 10, 20]
MACRO_LAGS     = [1, 5]
TARGET_HORIZON = 5
EPS = 1e-12

# -----------------------------------------
# Helpers
# -----------------------------------------
def read_auto(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        if df.shape[1] == 1 and ";" not in df.columns[0]:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    return df

def to_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    # dd.mm.yyyy sauber parsen
    try:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    except Exception:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)

def coerce_float(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Normalisierung von Feldnamen (macht vieles robust)
def norm_field(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    x = s.strip().lower()
    x = re.sub(r"\s+", "_", x)
    x = x.replace("%", "pct")
    aliases = {
        "ret": "return", "rtn": "return", "daily_return": "return", "px_return": "return",
        "pct_return": "return", "rendite": "return",
        "excess_return": "excess", "alpha": "excess",
        "adj_close": "adj_close", "adjclose": "adj_close", "adj._close": "adj_close",
        "close": "close", "schlusspreis": "close", "px_last": "close", "last": "close", "price": "close",
        "smi_ret": "smi_return", "smi": "smi", "smi_index_return": "smi_return",
    }
    return aliases.get(x, x)

def build_smi_return_from_df(smi_df: pd.DataFrame) -> pd.Series:
    cols = set(smi_df.columns)
    smi_df = to_datetime(smi_df.copy(), "Date")
    if {"Field", "Date", "Value"}.issubset(cols):
        smi_df["Field_norm"] = smi_df["Field"].map(norm_field)
        smi_df["Value"] = pd.to_numeric(smi_df["Value"], errors="coerce")
        wide = smi_df.pivot_table(index="Date", columns="Field_norm", values="Value", aggfunc="first").sort_index()
        if "smi_return" in wide.columns:
            return pd.to_numeric(wide["smi_return"], errors="coerce")
        for pc in ["close", "adj_close"]:
            if pc in wide.columns:
                p = pd.to_numeric(wide[pc], errors="coerce")
                return p.pct_change()
        raise ValueError("SMI data (long): weder 'SMI_Return' noch Preisspalten gefunden.")
    else:
        for rcol in ["SMI_Return", "Return"]:
            if rcol in smi_df.columns:
                return pd.to_numeric(smi_df.set_index("Date")[rcol], errors="coerce")
        for pc in ["Close", "Adj Close", "AdjClose", "Price", "PX_LAST"]:
            if pc in smi_df.columns:
                p = pd.to_numeric(smi_df.set_index("Date")[pc], errors="coerce")
                return p.pct_change()
        raise ValueError("SMI data (wide): weder 'SMI_Return'/'Return' noch Preisspalten gefunden.")

# -----------------------------------------
# Load
# -----------------------------------------
stocks_raw = read_auto(STOCKS_PATH)
macro_raw  = read_auto(MACRO_PATH)

need_stocks = {"Ticker", "Field", "Date", "Value"}
need_macro  = {"Field", "Date", "Value"}
if need_stocks - set(stocks_raw.columns):
    raise ValueError(f"stocks_long.csv fehlt Spalten: {need_stocks - set(stocks_raw.columns)}")
if need_macro - set(macro_raw.columns):
    raise ValueError(f"macro_long.csv fehlt Spalten: {need_macro - set(macro_raw.columns)}")

stocks_raw = to_datetime(stocks_raw, "Date")
macro_raw  = to_datetime(macro_raw,  "Date")
stocks_raw = coerce_float(stocks_raw, ["Value"])
macro_raw  = coerce_float(macro_raw,  ["Value"])

# Felder normalisieren
stocks_raw["Field_norm"] = stocks_raw["Field"].map(norm_field)
macro_raw["Field_norm"]  = macro_raw["Field"].map(norm_field)

# -----------------------------------------
# Pivot: Stocks -> (Date, Ticker) x Fields
# -----------------------------------------
stocks_wide = (
    stocks_raw.pivot_table(index=["Date","Ticker"], columns="Field_norm", values="Value", aggfunc="first")
    .sort_index()
)

# Falls kein Return vorhanden, aus Preisen pro Ticker ableiten
have_return = "return" in stocks_wide.columns
price_cols  = [c for c in ["close", "adj_close"] if c in stocks_wide.columns]

if not have_return and price_cols:
    # pro Ticker %Change
    df_list = []
    for t, g in stocks_wide.groupby(level="Ticker"):
        gi = g.droplevel("Ticker").copy().sort_index()
        price = None
        if "close" in gi.columns:    price = gi["close"]
        elif "adj_close" in gi.columns: price = gi["adj_close"]
        if price is not None:
            gi["return"] = pd.to_numeric(price, errors="coerce").pct_change()
        df_list.append(gi)
    stocks_wide = pd.concat({t: d for t, d in zip(stocks_wide.index.get_level_values("Ticker").unique(), df_list)}, names=["Ticker","Date"]).sort_index()

have_return = "return" in stocks_wide.columns
have_excess = "excess" in stocks_wide.columns

if not have_return and not have_excess:
    # Diagnosehilfe: welche Felder gibt es?
    sample_fields = sorted(list(stocks_raw["Field"].astype(str).unique()))[:20]
    raise ValueError(f"Weder 'Return' noch 'Excess' gefunden. Beispiel-Felder in stocks_long: {sample_fields}")

# -----------------------------------------
# Pivot: Macros -> Date x Fields
# -----------------------------------------
macro_wide = (
    macro_raw.pivot_table(index="Date", columns="Field_norm", values="Value", aggfunc="first")
    .sort_index()
)

# SMI_Return: erst aus Makros, sonst aus SMI-Datei
if "smi_return" in macro_wide.columns:
    smi_return = pd.to_numeric(macro_wide["smi_return"], errors="coerce")
else:
    smi_return = None
    if not have_excess:  # nur nötig, wenn wir Excess nicht direkt haben
        if not SMI_PATH.exists():
            raise ValueError("Kein 'Excess' vorhanden und SMI_Return weder in Makros noch als Datei verfügbar.")
        smi_df = read_auto(SMI_PATH)
        smi_return = build_smi_return_from_df(smi_df)

# Makro-Lags (nur Lags behalten -> keine zeitgleichen Makros)
macro_lags = pd.DataFrame(index=macro_wide.index)
for col in macro_wide.columns:
    for L in MACRO_LAGS:
        macro_lags[f"{col}_lag{L}"] = pd.to_numeric(macro_wide[col], errors="coerce").shift(L)

# -----------------------------------------
# Pro Ticker: Target & Features (leak-sicher)
# -----------------------------------------
frames = []

for t, g in stocks_wide.groupby(level="Ticker"):
    sub = g.droplevel("Ticker").copy().sort_index()  # Index=Date

    # Target: Summe t+1..t+5 (log-Excess)
    if have_excess:
        day_log_ex = np.log1p(pd.to_numeric(sub["excess"], errors="coerce") + EPS)
        target_log = sum(day_log_ex.shift(-h) for h in range(1, TARGET_HORIZON + 1))
        sub["Excess_5d_fwd"] = np.expm1(target_log)
    else:
        if smi_return is None:
            raise ValueError("Target kann nicht gebaut werden (weder 'excess' noch 'smi_return').")
        aligned = sub.join(smi_return.rename("smi_return"), how="left")
        day_log_ex = np.log1p(pd.to_numeric(aligned["return"], errors="coerce") + EPS) - \
                     np.log1p(pd.to_numeric(aligned["smi_return"], errors="coerce") + EPS)
        target_log = sum(day_log_ex.shift(-h) for h in range(1, TARGET_HORIZON + 1))
        sub["Excess_5d_fwd"] = np.expm1(target_log)

    # Rolling-Features (shift(1) = Info nur bis gestern)
    for w in ROLL_WINDOWS:
        if have_return:
            r = pd.to_numeric(sub["return"], errors="coerce")
            sub[f"Return_MA{w}"]  = r.rolling(w, min_periods=w).mean().shift(1)
            sub[f"Return_STD{w}"] = r.rolling(w, min_periods=w).std().shift(1)
        if have_excess:
            e = pd.to_numeric(sub["excess"], errors="coerce")
            sub[f"Excess_MA{w}"]  = e.rolling(w, min_periods=w).mean().shift(1)
            sub[f"Excess_STD{w}"] = e.rolling(w, min_periods=w).std().shift(1)

    # Lags
    if have_return:
        sub["Return_lag1"] = pd.to_numeric(sub["return"], errors="coerce").shift(1)
    if have_excess:
        sub["Excess_lag1"] = pd.to_numeric(sub["excess"], errors="coerce").shift(1)

    # Makro-Lags mergen
    sub = sub.merge(macro_lags, left_index=True, right_index=True, how="left")

    # Pflichtspalten dynamisch
    need_cols = ["Excess_5d_fwd"]
    for w in ROLL_WINDOWS:
        if have_return:
            need_cols += [f"Return_MA{w}", f"Return_STD{w}"]
        if have_excess:
            need_cols += [f"Excess_MA{w}", f"Excess_STD{w}"]
    if have_return: need_cols += ["Return_lag1"]
    if have_excess: need_cols += ["Excess_lag1"]

    sub = sub.dropna(subset=[c for c in need_cols if c in sub.columns]).copy()
    if sub.empty:
        continue

    sub.insert(0, "Date", sub.index)
    sub.insert(1, "Ticker", t)
    frames.append(sub.reset_index(drop=True))

if not frames:
    raise ValueError("Keine Datenzeilen nach Feature-Bau. Prüfe Eingaben/Feldnamen/NaNs.")

panel_long = pd.concat(frames, axis=0, ignore_index=True)

# Optional: Makrospalten mit hoher NaN-Quote entfernen
nan_ratio = panel_long.isna().mean()
drop_cols = nan_ratio[nan_ratio > 0.20].index.tolist()
if drop_cols:
    panel_long = panel_long.drop(columns=drop_cols)

panel_long = panel_long.sort_values(["Date","Ticker"]).reset_index(drop=True)
panel_long.to_csv(OUT_PATH, index=False)

print("Fertig.")
print("Tickers verarbeitet:", sorted(panel_long["Ticker"].unique().tolist()))
print("Zeilen (gesamt):", len(panel_long))
print("Export:", OUT_PATH.resolve())
print("Beispielspalten:", panel_long.columns[:24].tolist())
