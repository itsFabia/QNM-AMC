from __future__ import annotations
import os
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import csv


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling-Splits mit Embargo + RID-Guards erzeugen (robustes CSV-Parsing).")
    # IO
    p.add_argument("--csv", required=True, help="Pfad zur Input-CSV (Panel long).")
    p.add_argument("--out", required=True, help="Pfad zur splits_log.csv (bestimmt Ordner & Prefix).")
    # Columns (logische Namen; echte Namen werden automatisch aus dem Header gemappt)
    p.add_argument("--date-col", default="Date", help="Logischer Name der Datums-Spalte (Aliase erlaubt).")
    p.add_argument("--ticker-col", default="Ticker", help="Logischer Name der Ticker-Spalte (Aliase erlaubt).")
    p.add_argument("--price-col", default="Price", help="Logischer Name der Preis-Spalte (für optionalen Filter Price>0).")
    # Zeitfenster
    p.add_argument("--start", required=True, help="YYYY-MM-DD: ab diesem Monat beginnen die Testsplits.")
    p.add_argument("--train-years", type=int, default=3, help="Train-Fenster in Jahren.")
    p.add_argument("--test-months", type=int, default=1, help="Test-Fenster in Monaten.")
    p.add_argument("--embargo-days", type=int, default=5, help="Embargo in Kalendertagen um das Testfenster.")
    # Mindestgrößen
    p.add_argument("--min-train", type=int, default=250, help="Minimale Zeilenzahl im Train-Fenster.")
    p.add_argument("--min-test", type=int, default=20, help="Minimale Zeilenzahl im Test-Fenster.")
    # Optional: manuelles CSV-Parsing überschreiben
    p.add_argument("--sep", default=None, help="CSV-Trennzeichen erzwingen (z.B. ';').")
    p.add_argument("--encoding", default=None, help="Encoding (z.B. 'utf-8', 'latin-1').")
    return p.parse_args()


# -------------------- Utils --------------------

ALIASES = {
    "date":   {"date", "datum", "datetime", "timestamp"},
    "ticker": {"ticker", "ric", "symbol", "instrument", "id"},
    "price":  {"price", "close", "px_last", "last", "schlusskurs"},
}

def resolve_column(logical: str, df_cols: pd.Index) -> Optional[str]:
    """
    Mappe einen logischen Spaltennamen (z.B. 'Date') tolerant auf eine echte Spalte im DataFrame.
    - Ignoriert Groß/Kleinschreibung
    - Ignoriert Leerzeichen
    - Akzeptiert Aliase (s. ALIASES)
    """
    wanted = logical.strip().lower().replace(" ", "")
    cands = ALIASES.get(wanted, {wanted})
    for c in df_cols:
        key = str(c).strip().lower().replace(" ", "")
        if key in cands:
            return c
    return None


def autodetect_sep(path: str, encoding: Optional[str]) -> str:
    with open(path, "r", encoding=encoding or "utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            header = sample.splitlines()[0] if sample else ""
            # simple Heuristik: mehr ';' als ',' → ';' sonst ','
            return ';' if header.count(';') > header.count(',') else ','


def month_add(d: pd.Timestamp, months: int) -> pd.Timestamp:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, pd.Period(f"{y}-{m:02d}").days_in_month)
    return pd.Timestamp(year=y, month=m, day=day)


def make_rid(df: pd.DataFrame, date_col: str, ticker_col: str) -> np.ndarray:
    iso = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    key_df = pd.DataFrame({"d": iso, "t": df[ticker_col].astype("string")})
    rid = pd.util.hash_pandas_object(key_df, index=False, categorize=False).to_numpy(dtype="uint64")
    return rid


def build_splits(df: pd.DataFrame,
                 date_col: str,
                 start_date: str,
                 train_years: int,
                 test_months: int,
                 embargo_days: int,
                 min_train: int,
                 min_test: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    dts = pd.to_datetime(df[date_col], errors="coerce")
    if dts.isna().any():
        raise ValueError("Ungültige Datumswerte in DataFrame.")

    first = pd.Timestamp(start_date)
    last_day = dts.max().normalize()

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    cur_test_start = pd.Timestamp(first.year, first.month, 1)

    while cur_test_start <= last_day:
        cur_train_end = cur_test_start - pd.Timedelta(days=1)
        # Train-Start: gleiche Monatstagslogik (erster des Monats train_years früher)
        cur_train_start = pd.Timestamp(cur_train_end.year - train_years, cur_train_end.month, 1)

        cur_test_end = month_add(cur_test_start, test_months) - pd.Timedelta(days=1)

        mask_train_time = (dts >= cur_train_start) & (dts <= cur_train_end)
        mask_test_time  = (dts >= cur_test_start) & (dts <= cur_test_end)

        embargo_start = cur_test_start - pd.Timedelta(days=embargo_days)
        embargo_end   = cur_test_end + pd.Timedelta(days=embargo_days)
        mask_embargo = (dts >= embargo_start) & (dts <= embargo_end)

        mask_train = mask_train_time & (~mask_embargo)

        tr = np.flatnonzero(mask_train)
        te = np.flatnonzero(mask_test_time)

        if tr.size >= min_train and te.size >= min_test:
            splits.append((tr, te))

        cur_test_start = month_add(cur_test_start, 1)

    return splits


# -------------------- Main --------------------

def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # --- CSV robust laden ---
    sep = args.sep or autodetect_sep(args.csv, args.encoding)
    df = pd.read_csv(args.csv, sep=sep, encoding=args.encoding, engine="python")
    # Header normalisieren (nur Anzeige; wir behalten die Original-Bezeichner im DF)
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))

    # Pflichtspalten tolerant auflösen
    date_col_real   = resolve_column(args.date_col, df.columns)
    ticker_col_real = resolve_column(args.ticker_col, df.columns)
    price_col_real  = resolve_column(args.price_col, df.columns)

    if date_col_real is None or ticker_col_real is None:
        raise ValueError(f"Pflichtspalten fehlen. Gefunden: {list(df.columns)}")

    # Für weiteren Code die echten Namen verwenden
    args.date_col = date_col_real
    args.ticker_col = ticker_col_real
    if price_col_real is not None:
        args.price_col = price_col_real

    # Minimale, reproduzierbare Hygiene (auch später beim Scaling identisch ausführen!)
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col])
    if args.price_col in df.columns:
        # nur sinnvoll, wenn Preis existiert
        df = df[df[args.price_col] > 0]
    df = df.reset_index(drop=True)  # Positionsraum fixieren

    # RID-Guards (aus Date,Ticker)
    rid = make_rid(df, args.date_col, args.ticker_col)

    # Splits bauen
    splits = build_splits(df,
                          date_col=args.date_col,
                          start_date=args.start,
                          train_years=args.train_years,
                          test_months=args.test_months,
                          embargo_days=args.embargo_days,
                          min_train=args.min_train,
                          min_test=args.min_test)

    # Artefakt-Pfade
    base_no_ext = os.path.splitext(args.out)[0]
    idx_prefix = os.path.basename(base_no_ext) + "_idx"
    idx_dir = os.path.dirname(base_no_ext) or "."
    os.makedirs(idx_dir, exist_ok=True)

    # Schreiben
    rows = []
    for sid, (tr, te) in enumerate(splits):
        # Positions-Indizes
        pd.to_pickle(tr, os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_train.pkl"))
        pd.to_pickle(te, os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_test.pkl"))
        # RID-Guards
        np.save(os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_train_rid.npy"), rid[tr])
        np.save(os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_test_rid.npy"),  rid[te])

        rows.append({
            "split_id": sid,
            "train_start": df.loc[tr, args.date_col].min().date(),
            "train_end":   df.loc[tr, args.date_col].max().date(),
            "test_start":  df.loc[te, args.date_col].min().date(),
            "test_end":    df.loc[te, args.date_col].max().date(),
            "n_train": int(tr.size),
            "n_test":  int(te.size),
        })

    # Log schreiben
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Splits: {len(splits)}")
    print(f"[OK] Artefakte: {idx_dir} (Prefix: {idx_prefix})")
    print(f"[OK] Log: {args.out}")
    print(f"[INFO] CSV sep='{sep}', date_col='{args.date_col}', ticker_col='{args.ticker_col}', price_col='{args.price_col if args.price_col in df.columns else 'N/A'}'")


if __name__ == "__main__":
    main()
