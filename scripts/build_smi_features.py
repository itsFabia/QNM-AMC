# build_amc_features.py
# Erzeugt AMC-Features aus einem "wide" Panel (SMI-Titel + Makros).
# Alle Features sind strikt vergangenheitsbasiert (shift(1)), Target = Excess_5d_fwd (falls SMI erkannt).
#
# Voraussetzungen: pandas, numpy

import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path

def read_auto(path: Path) -> pd.DataFrame:
    """Robustes CSV-Loading (versucht ; und ,)."""
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        if df.shape[1] == 1 and ";" not in df.columns[0]:
            df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "date" in c.lower():
            return c
    raise ValueError("Keine Datums-Spalte gefunden. Kandidaten: %s" % list(df.columns)[:20])

def split_ticker(col: str) -> str:
    # "ABBN SE Equity | Last Price" -> "ABBN SE Equity"
    return col.split("|")[0].strip()

def engineer_equity_features(eq_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Pro Ticker Returns, Lags, Rolling, Z-Score etc. Alles past-only (shift(1))."""
    eq_df = eq_df.sort_values(["Ticker", date_col]).copy()
    eq_df["Return_raw"] = eq_df.groupby("Ticker")["Price"].pct_change()

    def past_only(g: pd.DataFrame) -> pd.DataFrame:
        r = g["Return_raw"]
        # Lags
        g["Return_lag1"] = r.shift(1)
        g["Return_lag5"] = r.shift(5)

        # Rolling Fenster -> danach shift(1) für past-only
        for w in [5, 10, 20]:
            ma = r.rolling(w, min_periods=max(3, w//2)).mean()
            sd = r.rolling(w, min_periods=max(3, w//2)).std(ddof=0)
            g[f"Return_MA{w}"]  = ma.shift(1)
            g[f"Return_STD{w}"] = sd.shift(1)

        # Annualisierte Vol
        g["RealizedVol20_ann"] = g["Return_STD20"] * np.sqrt(252)

        # Z-Score (Gegen Vergangenheitsfenster) -> shift(1)
        mu20  = r.rolling(20, min_periods=10).mean()
        std20 = r.rolling(20, min_periods=10).std(ddof=0)
        g["Return_Z20"] = ((r - mu20) / std20).shift(1)

        # Liquidity
        if "Volume" in g.columns:
            g["Volume_MA20"]   = g["Volume"].rolling(20, min_periods=10).mean().shift(1)
            g["Volume_to_MA20"] = (g["Volume"] / g["Volume_MA20"]).shift(1)

        # Fundamentals als Level -> nur laggen
        if "DivYld12m" in g.columns:
            g["DivYld12m_lag1"] = g["DivYld12m"].shift(1)
        if "MktCap" in g.columns:
            g["MktCap_lag1"] = g["MktCap"].shift(1)

        # Target-Kandidat (nicht shiften!)
        g["Ret_5d_fwd"] = g["Price"].pct_change(-5)
        return g

    eq_df = eq_df.groupby("Ticker", group_keys=False).apply(past_only)
    return eq_df

def add_cross_sectional_ranks(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ranks je Tag (bereits geshiftete Features verwenden)."""
    def rankit(col, name, ascending):
        if col in df.columns:
            df[name] = df.groupby(date_col)[col].rank(method="average", ascending=ascending, pct=True)
    rankit("Return_MA20", "Rank_MA20", ascending=False)     # hohes Momentum besser
    rankit("Return_STD20", "Rank_LowVol20", ascending=True) # niedrige Vol besser
    if "Volume_MA20" in df.columns:
        rankit("Volume_MA20", "Rank_Liquidity", ascending=False)
    return df

def build_macro_features(wide: pd.DataFrame, date_col: str, bench_price_col: str|None) -> pd.DataFrame:
    """Makros: nur past-only Ableitungen (Lag1, Lag5, logdiff1, chgstd20)."""
    macro_cols = [c for c in wide.columns if ("| Last Price" in c and "SE Equity" not in c)]
    if bench_price_col and bench_price_col in macro_cols:
        macro_cols.remove(bench_price_col)
    if not macro_cols:
        return pd.DataFrame({date_col: wide[date_col]})

    macro = wide[[date_col] + macro_cols].sort_values(date_col).copy()

    for c in macro_cols:
        macro[f"{c}__lag1"] = macro[c].shift(1)
        macro[f"{c}__lag5"] = macro[c].shift(5)
        # log-returns der Makroreihe, dann shift(1) -> past only
        logdiff1 = (np.log(macro[c]) - np.log(macro[c].shift(1))).shift(1)
        macro[f"{c}__logdiff1"] = logdiff1
        macro[f"{c}__chgstd20"] = (np.log(macro[c]) - np.log(macro[c].shift(1))).rolling(
            20, min_periods=10
        ).std(ddof=0).shift(1)

    keep = [date_col] + [c for c in macro.columns if "__" in c]
    return macro[keep]

def main(input_path: str, output_path: str):
    wide = read_auto(Path(input_path))
    date_col = find_date_col(wide)

    # Datum parsen
    wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    wide = wide.dropna(subset=[date_col]).sort_values(date_col)

    # Benchmark (SMI) erkennen
    bench_price_col = None
    for c in wide.columns:
        if c.upper().startswith("SMI") and "PX_LAST" in c.upper():
            bench_price_col = c
            break

    # Equity-Spalten sammeln
    price_cols = [c for c in wide.columns if ("SE Equity" in c and "| Last Price" in c)]
    vol_cols   = [c for c in wide.columns if ("SE Equity" in c and "| Volume" in c)]
    yld_cols   = [c for c in wide.columns if ("SE Equity" in c and "Dividend 12 Month Yld" in c)]
    mcap_cols  = [c for c in wide.columns if ("SE Equity" in c and "Current Market Cap" in c)]

    if not price_cols:
        raise ValueError("Keine Equity 'Last Price'-Spalten gefunden.")

    # Melt: Prices
    eq_price = wide[[date_col] + price_cols].melt(
        id_vars=[date_col], var_name="Instrument", value_name="Price"
    )
    eq_price["Ticker"] = eq_price["Instrument"].apply(split_ticker)
    eq_price = eq_price.drop(columns=["Instrument"])

    # Optional: Volume
    if vol_cols:
        eq_vol = wide[[date_col] + vol_cols].melt(
            id_vars=[date_col], var_name="Instrument", value_name="Volume"
        )
        eq_vol["Ticker"] = eq_vol["Instrument"].apply(split_ticker)
        eq_vol = eq_vol.drop(columns=["Instrument"])
        eq_price = eq_price.merge(eq_vol, on=[date_col, "Ticker"], how="left")

    # Optional: DivYld & MktCap (Levels werden später gelaggt)
    if yld_cols:
        eq_yld = wide[[date_col] + yld_cols].melt(
            id_vars=[date_col], var_name="Instrument", value_name="DivYld12m"
        )
        eq_yld["Ticker"] = eq_yld["Instrument"].apply(split_ticker)
        eq_yld = eq_yld.drop(columns=["Instrument"])
        eq_price = eq_price.merge(eq_yld, on=[date_col, "Ticker"], how="left")

    if mcap_cols:
        eq_mcap = wide[[date_col] + mcap_cols].melt(
            id_vars=[date_col], var_name="Instrument", value_name="MktCap"
        )
        eq_mcap["Ticker"] = eq_mcap["Instrument"].apply(split_ticker)
        eq_mcap = eq_mcap.drop(columns=["Instrument"])
        eq_price = eq_price.merge(eq_mcap, on=[date_col, "Ticker"], how="left")

    # Equity-Features (past-only)
    eq = engineer_equity_features(eq_price, date_col)

    # Benchmark-Returns + Target Excess_5d_fwd
    if bench_price_col:
        bench = wide[[date_col, bench_price_col]].rename(columns={bench_price_col: "SMI_Price"}).sort_values(date_col)
        bench["Bench_Return_raw"] = bench["SMI_Price"].pct_change()
        bench["Bench_Ret_5d_fwd"] = bench["SMI_Price"].pct_change(-5)
        eq = eq.merge(bench[[date_col, "Bench_Return_raw", "Bench_Ret_5d_fwd"]], on=date_col, how="left")
        # past-only Excess Feature
        eq["Excess_ret"] = (eq["Return_raw"] - eq["Bench_Return_raw"]).shift(1)
        # TARGET (nicht shiften)
        eq["Excess_5d_fwd"] = eq["Ret_5d_fwd"] - eq["Bench_Ret_5d_fwd"]

    # Cross-Sectional Ranks
    eq = add_cross_sectional_ranks(eq, date_col)

    # Makro-Features (past-only Ableitungen) und Merge by date
    macro_feat = build_macro_features(wide, date_col, bench_price_col)
    if not macro_feat.empty:
        eq = eq.merge(macro_feat, on=date_col, how="left")

    # Aufräumen
    for col in eq.columns:
        if eq[col].dtype.kind in "fc":
            eq[col] = eq[col].replace([np.inf, -np.inf], np.nan)

    eq = eq.sort_values(["Ticker", date_col])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(output_path, index=False)

    # Kurzer Hinweis
    kept = [
        date_col, "Ticker", "Price",
        "Return_lag1", "Return_lag5",
        "Return_MA5", "Return_STD5",
        "Return_MA10", "Return_STD10",
        "Return_MA20", "Return_STD20",
        "RealizedVol20_ann", "Return_Z20",
        "Volume", "Volume_MA20", "Volume_to_MA20",
        "DivYld12m_lag1", "MktCap_lag1",
        "Ret_5d_fwd", "Excess_5d_fwd" if "Excess_5d_fwd" in eq.columns else None,
        "Rank_MA20", "Rank_LowVol20", "Rank_Liquidity"
    ]
    kept = [c for c in kept if c and c in eq.columns]
    print("Fertig. Beispielspalten:", kept[:12])
    print(f"Zeilen: {len(eq):,}, Spalten: {eq.shape[1]} -> {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AMC Feature Builder (past-only, leakage-safe)")
    ap.add_argument("--input", required=True, help="Pfad zur Wide-Panel-CSV (z. B. FINAL_merged_SMI.csv)")
    ap.add_argument("--output", required=True, help="Pfad zur Output-CSV")
    args = ap.parse_args()
    main(args.input, args.output)
