from __future__ import annotations
import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# ---------- Helpers ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-Split Feature-Scaling (log1p + z-Score per Ticker).")
    p.add_argument("--csv", required=True, help="Pfad zur Input-CSV (Panel, Long-Format).")
    p.add_argument("--splits-log", required=True, help="Pfad zur splits_log.csv (um Prefix/Ordner abzuleiten).")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--ticker-col", default="Ticker")
    p.add_argument("--target", default="Excess_5d_fwd", help="Zielspalte (wird nicht skaliert).")
    p.add_argument("--price-col", default="Price", help="Optionaler Filter: Price>0 (falls vorhanden).")
    p.add_argument("--include-cols", default="", help="Komma-getrennt: explizite Featureliste (überschreibt Auto).")
    p.add_argument("--exclude-cols", default="", help="Komma-getrennt: Features ausschliessen (Regex ok).")
    p.add_argument("--auto-log1p", action="store_true", help="Automatisch log1p auf nicht-negative, stark schiefen Spalten.")
    p.add_argument("--log1p-cols", default="", help="Komma-getrennt: Spalten für log1p (zusätzlich zu auto).")
    p.add_argument("--impute-na", type=float, default=None, help="NaN-Imputation nach Z-Score (z.B. 0).")
    p.add_argument("--outdir", default="reports/scaled", help="Ausgabeordner.")
    return p.parse_args()


def infer_idx_pattern(splits_log_path: str) -> Tuple[str, str]:
    """
    Aus splits_log.csv den Ordner + Prefix der Pickles herleiten.
    train_test_split_rolling.py erzeugt Dateien wie:
      <dir>/<prefix>_000_train.pkl, <prefix>_000_test.pkl
    Beispiel: reports/splits_log.csv → Ordner 'reports', Prefix 'splits_log_idx'
    """
    base_no_ext = os.path.splitext(splits_log_path)[0]
    idx_prefix = os.path.basename(base_no_ext) + "_idx"
    idx_dir = os.path.dirname(base_no_ext) or "."
    return idx_dir, idx_prefix


def list_split_files(idx_dir: str, idx_prefix: str) -> List[int]:
    """
    Findet alle vorhandenen Split-IDs anhand der train/test-Pickle-Dateien.
    """
    ids = []
    if not os.path.isdir(idx_dir):
        return ids
    for fname in os.listdir(idx_dir):
        if fname.startswith(idx_prefix) and fname.endswith("_train.pkl"):
            # ..._NNN_train.pkl
            m = re.search(rf"{re.escape(idx_prefix)}_(\d+)_train\.pkl$", fname)
            if m:
                ids.append(int(m.group(1)))
    ids.sort()
    return ids


def auto_feature_cols(df: pd.DataFrame, date_col: str, ticker_col: str, target: str) -> List[str]:
    """
    Automatisch numerische Features erkennen (float/int) – Date, Ticker, Target raus.
    """
    exclude = {date_col, ticker_col, target}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def apply_log1p(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        # Nur anwenden, wenn alle Werte >= -1e-12 (numerische Toleranz)
        series = df[c]
        if series.min(skipna=True) >= -1e-12:
            df[c] = np.log1p(series.clip(lower=0))
        # Sonst überspringen wir still (kein Fehler), da log1p nicht definiert ist.


def choose_log1p_cols(df: pd.DataFrame, feature_cols: List[str],
                      auto: bool, manual: List[str]) -> List[str]:
    cand = set()
    if auto:
        # Heuristik: nicht-negativ & starke Schiefe
        for c in feature_cols:
            s = df[c]
            if s.dropna().empty:
                continue
            if s.min(skipna=True) >= 0:
                # Fisher-Pearson Schiefe
                skew = s.dropna().skew()
                if skew is not None and np.isfinite(skew) and skew > 1.0:
                    cand.add(c)
    for c in manual:
        if c and c in df.columns:
            cand.add(c)
    return sorted(cand)


def filter_features_by_regex(cols: List[str], exclude_patterns: List[str]) -> List[str]:
    if not exclude_patterns:
        return cols
    out = []
    for c in cols:
        if any(re.search(pat, c) for pat in exclude_patterns if pat):
            continue
        out.append(c)
    return out


def compute_group_stats(train_df: pd.DataFrame, ticker_col: str, cols: List[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Gibt dict[feature][ticker] = (mean, std) zurück.
    """
    stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    # Gruppiert nach Ticker einmal vorbereiten
    g = train_df.groupby(ticker_col, observed=True)
    for c in cols:
        s = g[c]
        means = s.mean()
        stds = s.std(ddof=0)  # Pop-Std; robust gegen kleine Gruppen
        stats[c] = {t: (float(means.get(t, np.nan)), float(stds.get(t, np.nan))) for t in means.index}
    return stats


def zscore_by_ticker(df_part: pd.DataFrame, ticker_col: str, cols: List[str],
                     stats: Dict[str, Dict[str, Tuple[float, float]]],
                     impute_na: Optional[float] = None) -> pd.DataFrame:
    """
    Wendet z-Score pro Ticker an: (x - mean_t) / std_t, std_t==0 -> 0.
    """
    df_out = df_part.copy()
    tickers = df_out[ticker_col].values
    for c in cols:
        if c not in df_out.columns:
            continue
        # Vektorisierte Map: baue Arrays der gleichen Länge
        means = np.array([stats[c].get(t, (np.nan, np.nan))[0] for t in tickers], dtype=float)
        stds  = np.array([stats[c].get(t, (np.nan, np.nan))[1] for t in tickers], dtype=float)
        vals  = df_out[c].to_numpy(dtype=float, copy=False)

        # z = (x - mu) / sigma; sigma==0 -> 0 (wenn x==mu), sonst 0 als neutral
        z = np.empty_like(vals)
        mask_valid = np.isfinite(vals) & np.isfinite(means) & np.isfinite(stds)
        z[:] = np.nan
        # std==0 separat
        zero_std = (stds == 0) & mask_valid
        z[zero_std] = 0.0
        # normalfall
        norm_mask = (stds != 0) & mask_valid
        z[norm_mask] = (vals[norm_mask] - means[norm_mask]) / stds[norm_mask]

        if impute_na is not None:
            # fehlende / unskalierbare Werte neutralisieren
            z[~np.isfinite(z)] = impute_na

        df_out[c] = z
    return df_out


# ---------- Main ----------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Daten laden
    df = pd.read_csv(args.csv)
    if args.date_col not in df.columns:
        raise ValueError(f"Spalte '{args.date_col}' fehlt.")
    if args.ticker_col not in df.columns:
        raise ValueError(f"Spalte '{args.ticker_col}' fehlt.")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col]).reset_index(drop=True)

    if args.price_col in df.columns:
        df = df[df[args.price_col] > 0].copy()

    # Spaltenauswahl
    include_cols = [c.strip() for c in args.include_cols.split(",")] if args.include_cols else []
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",")] if args.exclude_cols else []

    if include_cols:
        feature_cols = [c for c in include_cols if c in df.columns]
    else:
        feature_cols = auto_feature_cols(df, args.date_col, args.ticker_col, args.target)

    # Ziel/Schlüssel raus
    feature_cols = [c for c in feature_cols if c not in {args.date_col, args.ticker_col, args.target}]
    feature_cols = filter_features_by_regex(feature_cols, exclude_cols)
    if not feature_cols:
        raise RuntimeError("Keine Feature-Spalten gefunden. Prüfe include/exclude oder Datentypen.")

    # log1p-Auswahl & Anwendung (global, VOR Split-Stats)
    manual_log1p = [c.strip() for c in args.log1p_cols.split(",")] if args.log1p_cols else []
    log1p_cols = choose_log1p_cols(df, feature_cols, auto=args.auto_log1p, manual=manual_log1p)
    if log1p_cols:
        apply_log1p(df, log1p_cols)

    # Ordner/Prefix der Split-Indizes bestimmen
    idx_dir, idx_prefix = infer_idx_pattern(args.splits_log)
    split_ids = list_split_files(idx_dir, idx_prefix)
    if not split_ids:
        raise RuntimeError(f"Keine Split-Indizes gefunden unter {idx_dir} mit Prefix {idx_prefix}.")

    print(f"[INFO] Splits gefunden: {len(split_ids)} ({split_ids[0]}..{split_ids[-1]})")
    print(f"[INFO] Features ({len(feature_cols)}): {', '.join(feature_cols[:8])}{' ...' if len(feature_cols)>8 else ''}")
    if log1p_cols:
        print(f"[INFO] log1p auf: {', '.join(log1p_cols)}")
    if args.impute_na is not None:
        print(f"[INFO] NaN-Imputation gesetzt auf: {args.impute_na}")

    # Iteration über Splits
    for sid in split_ids:
        train_idx_path = os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_train.pkl")
        test_idx_path  = os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_test.pkl")
        train_idx = pd.read_pickle(train_idx_path).to_numpy()
        test_idx  = pd.read_pickle(test_idx_path).to_numpy()

        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        # Stats aus Train je Ticker
        stats = compute_group_stats(df_train, args.ticker_col, feature_cols)

        # z-Score anwenden (Train & Test)
        df_train_scaled = zscore_by_ticker(df_train, args.ticker_col, feature_cols, stats, args.impute_na)
        df_test_scaled  = zscore_by_ticker(df_test,  args.ticker_col, feature_cols, stats, args.impute_na)

        # Meta/Stats speichern (als Parquet für saubere Typen)
        stats_rows = []
        for c in feature_cols:
            for t, (m, s) in stats[c].items():
                stats_rows.append({"split_id": sid, "ticker": t, "feature": c, "mean": m, "std": s})
        stats_df = pd.DataFrame(stats_rows)

        # Output-Dateien
        out_train = os.path.join(args.outdir, f"scaled_train_{sid:03d}.parquet")
        out_test  = os.path.join(args.outdir, f"scaled_test_{sid:03d}.parquet")
        out_stats = os.path.join(args.outdir, f"scaler_stats_{sid:03d}.parquet")

        df_train_scaled.to_parquet(out_train, index=False)
        df_test_scaled.to_parquet(out_test, index=False)
        stats_df.to_parquet(out_stats, index=False)

        # kleine JSON-Summary (optional)
        meta = {
            "split_id": sid,
            "n_train": int(len(df_train_scaled)),
            "n_test": int(len(df_test_scaled)),
            "n_features": int(len(feature_cols)),
            "log1p_cols": log1p_cols,
            "impute_na": args.impute_na,
        }
        with open(os.path.join(args.outdir, f"meta_{sid:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[OK] Split {sid:03d} → train:{len(df_train_scaled)} test:{len(df_test_scaled)} "
              f"| saved: {os.path.basename(out_train)}, {os.path.basename(out_test)}, {os.path.basename(out_stats)}")

    print("[DONE] Scaling fertig.")


if __name__ == "__main__":
    main()
