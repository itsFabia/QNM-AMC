from __future__ import annotations
import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import csv


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-Split Scaling mit RID-Guards & robustem CSV-Parsing.")
    # IO
    p.add_argument("--csv", required=True)
    p.add_argument("--splits-log", required=True)
    p.add_argument("--outdir", default="reports/scaled")

    # Columns
    p.add_argument("--date-col", default="Date")
    p.add_argument("--ticker-col", default="Ticker")
    p.add_argument("--price-col", default="Price")

    # Target
    p.add_argument("--target", default="Excess_5d_fwd")
    p.add_argument("--build-target", action="store_true")
    p.add_argument("--bench-ret-col", default="Bench_Ret_5d_fwd")
    p.add_argument("--horizon", type=int, default=5)

    # Feature handling
    p.add_argument("--include-cols", default="")
    p.add_argument("--exclude-cols", default="")
    p.add_argument("--force-shift", action="store_true")

    # Scaling
    p.add_argument("--auto-log1p", action="store_true")
    p.add_argument("--log1p-cols", default="")
    p.add_argument("--impute-na", type=float, default=None)

    # CSV parsing
    p.add_argument("--sep", default=None)
    p.add_argument("--encoding", default=None)

    return p.parse_args()


# ---------- Helpers ----------
def autodetect_sep(path: str, encoding: Optional[str]) -> str:
    with open(path, "r", encoding=encoding or "utf-8", errors="ignore") as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            header = sample.splitlines()[0] if sample else ""
            return ";" if header.count(";") > header.count(",") else ","


def infer_idx_pattern(splits_log_path: str) -> Tuple[str, str]:
    base_no_ext = os.path.splitext(splits_log_path)[0]
    idx_prefix = os.path.basename(base_no_ext) + "_idx"
    idx_dir = os.path.dirname(base_no_ext) or "."
    return idx_dir, idx_prefix


def list_split_ids(idx_dir: str, idx_prefix: str) -> List[int]:
    ids = []
    for fn in os.listdir(idx_dir):
        if fn.startswith(idx_prefix) and fn.endswith("_train.pkl"):
            sid = int(fn.replace(f"{idx_prefix}_", "").replace("_train.pkl", ""))
            ids.append(sid)
    ids.sort()
    return ids


def make_rid(df: pd.DataFrame, date_col: str, ticker_col: str) -> np.ndarray:
    iso = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    key_df = pd.DataFrame({"d": iso, "t": df[ticker_col].astype("string")})
    return pd.util.hash_pandas_object(key_df, index=False, categorize=False).to_numpy(dtype="uint64")


def auto_feature_cols(df: pd.DataFrame, date_col: str, ticker_col: str, target: str) -> List[str]:
    excl = {date_col, ticker_col, target}
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def filter_features_by_regex(cols: List[str], exclude_patterns: List[str]) -> List[str]:
    if not exclude_patterns:
        return cols
    return [c for c in cols if not any(re.search(p, c) for p in exclude_patterns if p)]


def choose_log1p_cols(df: pd.DataFrame, feature_cols: List[str], auto: bool, manual: List[str]) -> List[str]:
    cand = set()
    if auto:
        for c in feature_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            if s.min() >= 0 and np.isfinite(s.skew()) and s.skew() > 1.0:
                cand.add(c)
    for c in manual:
        if c and c in df.columns:
            cand.add(c)
    return sorted(cand)


def apply_log1p_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns and df[c].min(skipna=True) >= -1e-12:
            df[c] = np.log1p(df[c].clip(lower=0))


def compute_group_stats(train_df: pd.DataFrame, ticker_col: str, cols: List[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    g = train_df.groupby(ticker_col, observed=True)
    for c in cols:
        means = g[c].mean()
        stds = g[c].std(ddof=0)
        stats[c] = {t: (float(means.get(t, np.nan)), float(stds.get(t, np.nan))) for t in means.index}
    return stats


def zscore_by_ticker(df_part: pd.DataFrame, ticker_col: str, cols: List[str],
                     stats: Dict[str, Dict[str, Tuple[float, float]]],
                     impute_na: Optional[float]) -> pd.DataFrame:
    out = df_part.copy()
    tickers = out[ticker_col].to_numpy()
    for c in cols:
        if c not in out.columns:
            continue
        vals = out[c].to_numpy(dtype=float, copy=False)
        means = np.array([stats[c].get(t, (np.nan, np.nan))[0] for t in tickers], dtype=float)
        stds  = np.array([stats[c].get(t, (np.nan, np.nan))[1] for t in tickers], dtype=float)
        z = np.full_like(vals, np.nan)
        mask = np.isfinite(vals) & np.isfinite(means) & np.isfinite(stds)
        z[(stds == 0) & mask] = 0.0
        nz = (stds != 0) & mask
        z[nz] = (vals[nz] - means[nz]) / stds[nz]
        if impute_na is not None:
            z[~np.isfinite(z)] = impute_na
        out[c] = z
    return out


def build_targets_if_needed(df: pd.DataFrame, date_col: str, ticker_col: str,
                            price_col: str, bench_ret_col: str, horizon: int,
                            target_col: str) -> pd.DataFrame:
    if target_col in df.columns:
        return df
    if price_col not in df.columns:
        raise ValueError(f"--build-target verlangt '{price_col}'.")

    def forward_ret(g: pd.DataFrame) -> pd.Series:
        ord_ = np.argsort(g[date_col].values, kind="mergesort")
        inv = np.empty_like(ord_)
        inv[ord_] = np.arange(len(ord_))
        p = g[price_col].to_numpy()
        p_sorted = p[ord_]
        r_sorted = pd.Series(p_sorted).pct_change(horizon).shift(-horizon).to_numpy()
        r = np.full_like(p_sorted, np.nan, dtype=float)
        r[inv] = r_sorted
        return pd.Series(r, index=g.index)

    df = df.copy()
    ret_col = f"Ret_{horizon}d_fwd"
    df[ret_col] = df.groupby(ticker_col, observed=True, group_keys=False).apply(forward_ret)
    if bench_ret_col not in df.columns:
        df[bench_ret_col] = 0.0
    df[target_col] = df[ret_col] - df[bench_ret_col]
    return df


def force_shift_features_tminus1(df: pd.DataFrame, feature_cols: List[str], ticker_col: str) -> pd.DataFrame:
    if not feature_cols:
        return df
    df = df.copy()
    g = df.groupby(ticker_col, observed=True)
    for c in feature_cols:
        df[c] = g[c].shift(1)
    return df


# ---------- Main ----------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # CSV laden (robust)
    sep = args.sep or autodetect_sep(args.csv, args.encoding)
    df = pd.read_csv(args.csv, sep=sep, encoding=args.encoding, engine="python")
    df.columns = df.columns.astype(str).str.strip()
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[args.date_col])
    if args.price_col in df.columns:
        df = df[df[args.price_col] > 0]
    df = df.reset_index(drop=True)

    # Optional Target bauen
    if args.build_target:
        df = build_targets_if_needed(df, args.date_col, args.ticker_col,
                                     args.price_col, args.bench_ret_col,
                                     args.horizon, args.target)

    # Feature-Auswahl
    include_cols = [c.strip() for c in args.include_cols.split(",") if c.strip()]
    exclude_pats = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    feature_cols = [c for c in include_cols if c in df.columns] if include_cols else \
                   auto_feature_cols(df, args.date_col, args.ticker_col, args.target)
    feature_cols = filter_features_by_regex([c for c in feature_cols if c not in {args.date_col, args.ticker_col, args.target}], exclude_pats)
    if not feature_cols:
        raise RuntimeError("Keine Feature-Spalten gefunden.")

    if args.force_shift:
        df = force_shift_features_tminus1(df, feature_cols, args.ticker_col)

    manual_log1p = [c.strip() for c in args.log1p_cols.split(",") if c.strip()]
    log1p_cols = choose_log1p_cols(df, feature_cols, auto=args.auto_log1p, manual=manual_log1p)
    if log1p_cols:
        apply_log1p_inplace(df, log1p_cols)

    # Splits
    idx_dir, idx_prefix = infer_idx_pattern(args.splits_log)
    split_ids = list_split_ids(idx_dir, idx_prefix)
    if not split_ids:
        raise RuntimeError(f"Keine Splits gefunden unter {idx_dir} mit Prefix {idx_prefix}.")
    df_rid = make_rid(df, args.date_col, args.ticker_col)

    print(f"[INFO] Splits: {len(split_ids)} | Features: {len(feature_cols)}")

    for sid in split_ids:
        tr_path = os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_train.pkl")
        te_path = os.path.join(idx_dir, f"{idx_prefix}_{sid:03d}_test.pkl")
        tr_rid_path = tr_path.replace("_train.pkl", "_train_rid.npy")
        te_rid_path = te_path.replace("_test.pkl", "_test_rid.npy")

        train_idx = pd.read_pickle(tr_path)
        test_idx = pd.read_pickle(te_path)

        n = len(df)
        if train_idx.max() >= n or test_idx.max() >= n:
            raise RuntimeError(f"Split {sid}: Index außerhalb Range (n={n}).")

        if os.path.exists(tr_rid_path) and os.path.exists(te_rid_path):
            rid_tr = np.load(tr_rid_path)
            rid_te = np.load(te_rid_path)
            if not (np.array_equal(rid_tr, df_rid[train_idx]) and np.array_equal(rid_te, df_rid[test_idx])):
                raise RuntimeError(f"Split {sid}: RID-Check fehlgeschlagen. Splits neu generieren.")

        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        df_train = df_train.sort_values([args.date_col, args.ticker_col]).reset_index(drop=True)
        df_test  = df_test.sort_values([args.date_col, args.ticker_col]).reset_index(drop=True)

        if args.target in df_train.columns:
            df_train = df_train.dropna(subset=[args.target])
        if args.target in df_test.columns:
            df_test = df_test.dropna(subset=[args.target])

        stats = compute_group_stats(df_train, args.ticker_col, feature_cols)
        df_train_scaled = zscore_by_ticker(df_train, args.ticker_col, feature_cols, stats, args.impute_na)
        df_test_scaled  = zscore_by_ticker(df_test,  args.ticker_col, feature_cols, stats, args.impute_na)

        out_train = os.path.join(args.outdir, f"scaled_train_{sid:03d}.parquet")
        out_test  = os.path.join(args.outdir, f"scaled_test_{sid:03d}.parquet")
        out_stats = os.path.join(args.outdir, f"scaler_stats_{sid:03d}.parquet")

        stats_rows = []
        for c, mp in stats.items():
            for t, (m, s) in mp.items():
                stats_rows.append({"split_id": sid, "ticker": t, "feature": c, "mean": m, "std": s})
        stats_df = pd.DataFrame(stats_rows)

        df_train_scaled.to_parquet(out_train, index=False)
        df_test_scaled.to_parquet(out_test, index=False)
        stats_df.to_parquet(out_stats, index=False)

        meta = {
            "split_id": sid,
            "n_train": len(df_train_scaled),
            "n_test": len(df_test_scaled),
            "features": feature_cols,
            "log1p_cols": log1p_cols,
            "impute_na": args.impute_na,
            "force_shift": bool(args.force_shift),
            "target": args.target,
        }
        with open(os.path.join(args.outdir, f"meta_{sid:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[OK] Split {sid:03d}: train={len(df_train_scaled)}, test={len(df_test_scaled)}")

    print("[DONE] Scaling fertig.")


if __name__ == "__main__":
    main()
