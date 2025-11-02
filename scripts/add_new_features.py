"""
Erweitert AMC_model_input.csv um neue Features.
Liest die existierenden Daten und fügt die neuen Feature-Kategorien hinzu.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt neue Features zu DataFrame hinzu (per Ticker)."""

    print(f"Eingabe: {len(df)} Zeilen, {df.shape[1]} Spalten")

    def engineer_per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()

        # Check if we have the required columns
        return_col = None
        for col in ["Return_raw", "return", "Return"]:
            if col in g.columns:
                return_col = col
                break

        if return_col is None:
            print(f"  Warnung: Keine Return-Spalte gefunden")
            return g

        r = g[return_col]

        # === MULTI-TIMEFRAME MOMENTUM ===
        for period, name in [(21, "1M"), (63, "3M"), (126, "6M")]:
            cum_ret = (1 + r).rolling(period, min_periods=period//2).apply(
                lambda x: x.prod() - 1, raw=True
            )
            g[f"Return_{name}"] = cum_ret.shift(1)

        # === MEAN REVERSION ===
        # RSI-like
        up = r.clip(lower=0).rolling(20, min_periods=10).sum()
        down = (-r.clip(upper=0)).rolling(20, min_periods=10).sum()
        rsi = 100 * up / (up + down + 1e-10)
        g["RSI20"] = rsi.shift(1)

        # Distance to MA (if we have price data)
        price_col = None
        for col in ["Price", "close", "px_last", "price"]:
            if col in g.columns:
                price_col = col
                break

        if price_col:
            p = g[price_col]
            for w in [10, 20, 60]:
                ma = p.rolling(w, min_periods=w//2).mean()
                distance = ((p - ma) / ma).shift(1)
                g[f"DistMA{w}"] = distance

        # === TREND CONSISTENCY ===
        for w in [10, 20]:
            pct_positive = (r > 0).rolling(w, min_periods=w//2).mean()
            g[f"PctPos{w}d"] = pct_positive.shift(1)

        # === MOMENTUM ACCELERATION ===
        ma10 = r.rolling(10, min_periods=5).mean()
        ma20 = r.rolling(20, min_periods=10).mean()
        mom_accel = (ma10 - ma20).shift(1)
        g["MomAccel"] = mom_accel

        # === VOLATILITY-ADJUSTED RETURNS ===
        vol_adj_ret = (r / (r.rolling(20, min_periods=10).std(ddof=0) + 1e-10)).shift(1)
        g["VolAdjRet20"] = vol_adj_ret

        # === VOLATILITY REGIME ===
        vol = r.rolling(20, min_periods=10).std(ddof=0)
        vol_of_vol = vol.rolling(20, min_periods=10).std(ddof=0).shift(1)
        g["VolOfVol"] = vol_of_vol

        # === PRICE-VOLUME FEATURES ===
        volume_col = None
        for col in ["Volume", "volume", "VOLUME"]:
            if col in g.columns:
                volume_col = col
                break

        if volume_col:
            vol_pct_change = g[volume_col].pct_change()
            # Price-Volume Correlation
            corr_window = 20
            pv_corr = r.rolling(corr_window, min_periods=corr_window//2).corr(
                vol_pct_change
            ).shift(1)
            g["PriceVol_Corr"] = pv_corr

            # Volume Trend
            vol_ma10 = g[volume_col].rolling(10, min_periods=5).mean()
            vol_ma20 = g[volume_col].rolling(20, min_periods=10).mean()
            g["VolTrend"] = (vol_ma10 / vol_ma20).shift(1)

        return g

    # Apply per ticker
    df_enhanced = df.groupby("Ticker", group_keys=False).apply(engineer_per_ticker)

    print(f"Nach Feature-Engineering: {df_enhanced.shape[1]} Spalten")

    # Add cross-sectional features
    print("Füge Cross-Sectional Features hinzu...")

    # Rankings
    for feat, ascending in [
        ("Return_1M", False),
        ("Return_3M", False),
        ("Return_6M", False),
        ("RSI20", False),
        ("VolAdjRet20", False),
        ("PctPos20d", False)
    ]:
        if feat in df_enhanced.columns:
            df_enhanced[f"Rank_{feat}"] = df_enhanced.groupby("Date")[feat].rank(
                method="average", ascending=ascending, pct=True
            )

    # Cross-sectional spreads
    for feat in ["Return_1M", "Return_3M", "VolAdjRet20", "RSI20"]:
        if feat in df_enhanced.columns:
            cs_mean = df_enhanced.groupby("Date")[feat].transform("mean")
            cs_std = df_enhanced.groupby("Date")[feat].transform("std")
            df_enhanced[f"{feat}_CSSpread"] = (df_enhanced[feat] - cs_mean) / (cs_std + 1e-10)

    print(f"Final: {df_enhanced.shape[1]} Spalten")

    # Clean up infinities
    for col in df_enhanced.columns:
        if df_enhanced[col].dtype.kind in "fc":
            df_enhanced[col] = df_enhanced[col].replace([np.inf, -np.inf], np.nan)

    return df_enhanced


def main():
    input_path = Path("data/AMC_model_input.csv")
    output_path = Path("data/AMC_model_input.csv")
    backup_path = Path("data/AMC_model_input_OLD_FEATURES.csv")

    print(f"Lese {input_path}...")

    # Read with proper separator detection
    try:
        df = pd.read_csv(input_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(input_path)
    except:
        df = pd.read_csv(input_path)

    # Backup original
    if backup_path.exists():
        backup_path.unlink()
    df.to_csv(backup_path, index=False)
    print(f"Backup erstellt: {backup_path}")

    # Backup Date column before conversion
    date_col_backup = df["Date"].copy()

    # Ensure Date is datetime for feature engineering
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    # Add new features
    df_new = add_new_features(df)

    # Restore original Date format for saving
    df_new["Date"] = date_col_backup

    # Save
    df_new.to_csv(output_path, index=False)
    print(f"\nGespeichert: {output_path}")
    print(f"Neue Feature-Anzahl: {df_new.shape[1] - df.shape[1]}")

    new_features = [c for c in df_new.columns if c not in df.columns]
    print(f"\nNeue Features ({len(new_features)}):")
    for f in sorted(new_features)[:20]:
        print(f"  - {f}")
    if len(new_features) > 20:
        print(f"  ... und {len(new_features) - 20} weitere")


if __name__ == "__main__":
    main()
