# -*- coding: utf-8 -*-
"""
Rolling/Expanding Window Cross-Validation mit Embargo
=====================================================

Ziele:
- Zeitreihen-sichere Splits (kein Leakage) mit Embargo-Puffer.
- Faire Startbasis (z. B. 2021-01-01) trotz später IPOs.
- Saubere Validierung (Spalten, Cross-Section, Datumsbereich).
- Cold-Start-Policy (Score erst ab N ≥ 250 im Train-Split).
- Split-Logging inkl. Ticker-Verteilung und Export der Split-Indizes.

CLI (Beispiel):
    python train_test_split_rolling.py ^
        --csv C:\Users\holze\PycharmProjects\QNM-AMC\data\AMC_model_input_reduced.csv ^
        --start 2021-01-01 ^
        --train-years 3 ^
        --test-months 1 ^
        --embargo-days 5 ^
        --mode rolling ^
        --date-col Date ^
        --price-col Price ^
        --out C:\Users\holze\PycharmProjects\QNM-AMC\reports\splits_log.csv ^
        --min-train-days-per-ticker 250
"""

from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Literal
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


# ---------- Utilities ----------

def detect_ticker_column(df: pd.DataFrame, candidates=None) -> str:
    if candidates is None:
        candidates = ["Ticker", "Instrument", "Symbol", "RIC", "Name"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    nunique = df.nunique(dropna=False)
    exclude = {"date", "datetime", "time"}
    textish = [c for c in df.columns if (c.lower() not in exclude and df[c].dtype == object)]
    if textish:
        picks = [c for c in textish if 2 <= nunique[c] <= 2000]
        if picks:
            return picks[0]
    raise ValueError("Ticker-Spalte nicht gefunden. Bitte --ticker-col explizit setzen.")


def monthly_starts(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DatetimeIndex:
    lo = min_date.normalize().replace(day=1)
    hi = max_date.normalize().replace(day=1)
    return pd.date_range(lo, hi, freq="MS")


@dataclass
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: pd.Index | np.ndarray
    test_idx: pd.Index | np.ndarray

    def meta(self, df: pd.DataFrame, date_col: str, ticker_col: str) -> Dict[str, Any]:
        # positionsbasiert!
        train = df.iloc[self.train_idx].copy()
        test = df.iloc[self.test_idx].copy()
        train["__day"] = pd.to_datetime(train[date_col]).dt.normalize()
        test["__day"] = pd.to_datetime(test[date_col]).dt.normalize()

        train_counts = train[ticker_col].value_counts(normalize=True).rename(lambda x: f"train_share__{x}")
        test_counts = test[ticker_col].value_counts(normalize=True).rename(lambda x: f"test_share__{x}")

        base = dict(
            train_start=self.train_start.date(),
            train_end=self.train_end.date(),
            test_start=self.test_start.date(),
            test_end=self.test_end.date(),
            train_rows=len(train),
            test_rows=len(test),
            train_days=int(train["__day"].nunique()),
            test_days=int(test["__day"].nunique()),
            train_date_min=train["__day"].min().date() if len(train) else None,
            train_date_max=train["__day"].max().date() if len(train) else None,
            test_date_min=test["__day"].min().date() if len(test) else None,
            test_date_max=test["__day"].max().date() if len(test) else None,
        )
        base.update(train_counts.to_dict())
        base.update(test_counts.to_dict())
        return base


def make_splits(
    df: pd.DataFrame,
    *,
    start: pd.Timestamp,
    train_years: int,
    test_months: int,
    embargo_days: int,
    date_col: str,
    ticker_col: str,
    mode: Literal["rolling", "expanding"] = "rolling",
) -> List[Split]:
    """
    Erwartet df bereits sortiert nach [date_col, ticker_col] mit reset_index(drop=True)!
    Regeln:
      - TrainEnd = TestStart - Embargo
      - TrainStart:
          rolling:   TrainEnd - train_years
          expanding: global_min_date .. TrainEnd
      - Test: [TestStart, TestStart + test_months)
    """
    gmin, gmax = df[date_col].min(), df[date_col].max()
    test_starts_all = monthly_starts(gmin, gmax)
    test_starts = test_starts_all[test_starts_all >= start]

    out: List[Split] = []
    for ts in test_starts:
        train_end = (ts - pd.Timedelta(days=embargo_days)).normalize()
        if train_end <= gmin:
            continue

        if mode == "rolling":
            train_start = (train_end - DateOffset(years=train_years)).normalize()
        else:
            train_start = gmin.normalize()

        test_end = (ts + DateOffset(months=test_months)).normalize()

        train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
        test_mask = (df[date_col] >= ts) & (df[date_col] < test_end)
        if not test_mask.any():
            continue

        split = Split(
            train_start=train_start, train_end=train_end,
            test_start=ts, test_end=test_end,
            train_idx=df.index[train_mask],
            test_idx=df.index[test_mask],
        )

        # Safety: max(TrainDate) < TestStart
        if df.iloc[split.train_idx][date_col].max() >= ts:
            continue

        out.append(split)
    return out


def summarize_cross_section(df: pd.DataFrame, date_col: str, ticker_col: str) -> Dict[str, Any]:
    day = pd.to_datetime(df[date_col]).dt.normalize()
    per_day = df.groupby(day)[ticker_col].nunique()
    return dict(
        unique_ticker=int(df[ticker_col].nunique()),
        avg_ticker_per_day=float(per_day.mean()),
        min_ticker_per_day=int(per_day.min()),
        max_ticker_per_day=int(per_day.max())
    )


def write_log(df: pd.DataFrame, splits: List[Split], out_csv: str, date_col: str, ticker_col: str):
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rows = [s.meta(df, date_col=date_col, ticker_col=ticker_col) for s in splits]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Splits-Log geschrieben: {out_csv}  (Splits: {len(splits)})")


def save_split_indices(splits: List[Split], out_dir: str, prefix="split"):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i, s in enumerate(splits):
        pd.Series(s.train_idx, name="train_idx").to_pickle(os.path.join(out_dir, f"{prefix}_{i:03d}_train.pkl"))
        pd.Series(s.test_idx,  name="test_idx").to_pickle(os.path.join(out_dir, f"{prefix}_{i:03d}_test.pkl"))
    print(f"[OK] Indizes gespeichert unter: {out_dir}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling/Expanding CV mit Embargo (zeitreihensicher).")
    p.add_argument("--csv", required=True)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--embargo-days", type=int, default=5)
    p.add_argument("--mode", choices=["rolling", "expanding"], default="rolling")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--ticker-col", default=None)
    p.add_argument("--price-col", default="Price")
    p.add_argument("--out", default="reports/splits_log.csv")
    p.add_argument("--min-train-days-per-ticker", type=int, default=250)
    p.add_argument("--strict-coldstart", action="store_true",
                   help="Wenn gesetzt: Splits ohne ausreichende Train-Historie werden verworfen.")
    return p.parse_args()


def main():
    args = parse_args()

    # Einlesen
    df = pd.read_csv(args.csv)
    if args.date_col not in df.columns:
        raise ValueError(f"Spalte '{args.date_col}' fehlt in {args.csv}.")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col]).reset_index(drop=True)

    # Ticker-Spalte
    ticker_col = args.ticker_col or detect_ticker_column(df)
    if ticker_col not in df.columns:
        raise ValueError(f"Ticker-Spalte '{ticker_col}' nicht gefunden.")

    # Optionaler Price>0 Filter
    if args.price_col in df.columns:
        before = len(df)
        df = df[df[args.price_col] > 0].copy()
        print(f"[INFO] Price-Filter '{args.price_col} > 0' angewendet: {before} → {len(df)} Zeilen.")

    # Einheitliche Sortierung für ALLE nachfolgenden Indizes
    df = df.sort_values([args.date_col, ticker_col]).reset_index(drop=True)
    print("[INFO] Sortierung gesetzt: [Date, Ticker] + reset_index(drop=True)")

    # Grundlegende Infos
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print("Date range:", df[args.date_col].min().date(), "→", df[args.date_col].max().date())
    print("Ticker column:", ticker_col)

    # Cross-Section Check
    cs = summarize_cross_section(df, date_col=args.date_col, ticker_col=ticker_col)
    print(f"Unique Ticker: {cs['unique_ticker']}")
    print(f"Ø Ticker/Tag: {cs['avg_ticker_per_day']:.2f} (min {cs['min_ticker_per_day']}, max {cs['max_ticker_per_day']})")

    # Splits bauen (df ist bereits korrekt sortiert)
    splits = make_splits(
        df=df,
        start=pd.to_datetime(args.start),
        train_years=args.train_years,
        test_months=args.test_months,
        embargo_days=args.embargo_days,
        date_col=args.date_col,
        ticker_col=ticker_col,
        mode=args.mode,
    )

    # Cold-Start-Policy anwenden
    eligible_splits: List[Split] = []
    for s in splits:
        train_df = df.iloc[s.train_idx][[args.date_col, ticker_col]].copy()
        train_df["__day"] = pd.to_datetime(train_df[args.date_col]).dt.normalize()
        counts = train_df.groupby(ticker_col)["__day"].nunique()
        ok_ticker = set(counts[counts >= args.min_train_days_per_ticker].index)

        if args.strict_coldstart and len(ok_ticker) == 0:
            continue

        test_df = df.iloc[s.test_idx][[ticker_col]].copy()
        keep_mask = test_df[ticker_col].isin(ok_ticker).values
        if keep_mask.sum() == 0 and args.strict_coldstart:
            continue

        filtered_test_idx = np.array(s.test_idx)[keep_mask]
        s = Split(
            train_start=s.train_start, train_end=s.train_end,
            test_start=s.test_start, test_end=s.test_end,
            train_idx=s.train_idx,
            test_idx=filtered_test_idx,
        )
        eligible_splits.append(s)

    splits = eligible_splits
    if not splits:
        raise RuntimeError("Alle Splits fielen durch die Cold-Start-Policy.")

    # Logging + Index-Export
    write_log(df, splits, args.out, date_col=args.date_col, ticker_col=ticker_col)
    idx_dir = os.path.splitext(args.out)[0] + "_idx"
    out_dir = os.path.dirname(idx_dir) or "."
    prefix = os.path.basename(idx_dir)
    save_split_indices(splits, out_dir, prefix)

    # Beispielausgabe
    first = splits[0]
    meta = first.meta(df, date_col=args.date_col, ticker_col=ticker_col)
    print("\nBeispiel-Split:")
    for k in ["train_start", "train_end", "test_start", "test_end",
              "train_rows", "test_rows", "train_days", "test_days",
              "train_date_min", "train_date_max", "test_date_min", "test_date_max"]:
        print(f"  {k}: {meta[k]}")
    train_shares = {k.replace("train_share__", ""): v for k, v in meta.items() if k.startswith("train_share__")}
    test_shares = {k.replace("test_share__", ""): v for k, v in meta.items() if k.startswith("test_share__")}
    train_top = sorted(train_shares.items(), key=lambda kv: kv[1], reverse=True)[:10]
    test_top = sorted(test_shares.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTicker-Anteile im Beispiel-Split (Top 10 train/test):")
    print("  Train:", ", ".join([f"{t}:{p:.3f}" for t, p in train_top]))
    print("  Test :", ", ".join([f"{t}:{p:.3f}" for t, p in test_top]))


if __name__ == "__main__":
    main()

