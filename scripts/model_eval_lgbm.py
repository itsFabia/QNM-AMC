# scripts/model_eval_lgbm.py
# QNM AMC – robuste Modell-Evaluation (LGBM Regressor + optional LambdaRank) über Rolling-Splits
from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMRanker
from scipy.stats import spearmanr


# =========================
# Logging & Utils
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")


def infer_feature_cols(df: pd.DataFrame, target: str, drop_cols: List[str]) -> List[str]:
    bad = set([c for c in drop_cols if c in df.columns] + [target])
    feats = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def drop_constant_and_inf(df: pd.DataFrame) -> pd.DataFrame:
    # drop inf/-inf cols
    num = df.select_dtypes(include=[np.number])
    is_inf = np.isinf(num).any()
    bad_inf = list(is_inf[is_inf].index)
    df = df.drop(columns=bad_inf, errors="ignore")

    # drop constant cols
    num2 = df.select_dtypes(include=[np.number])
    nunique = num2.nunique(dropna=False)
    bad_const = list(nunique[nunique <= 1].index)
    df = df.drop(columns=bad_const, errors="ignore")
    return df


def make_val_split_by_time(train: pd.DataFrame, date_col: str, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates_sorted = np.sort(train[date_col].dropna().unique())
    if len(dates_sorted) < 4:
        # Fallback: random
        val = train.sample(frac=val_frac, random_state=42)
        trn = train.drop(val.index)
        return trn, val
    split_idx = int(np.floor((1 - val_frac) * len(dates_sorted)))
    split_idx = min(max(split_idx, 1), len(dates_sorted) - 1)
    cutoff = dates_sorted[split_idx - 1]
    trn = train[train[date_col] <= cutoff].copy()
    val = train[train[date_col] > cutoff].copy()
    if len(val) < 50 or len(trn) < 200:
        val = train.sample(frac=val_frac, random_state=42)
        trn = train.drop(val.index)
    return trn, val


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan
    r, _ = spearmanr(y_true, y_pred)
    return r


def daily_metrics(df: pd.DataFrame, date_col: str, target: str, pred_col: str, topk: int) -> Dict[str, float]:
    ics, hits, ndays = [], [], 0
    for _, g in df.groupby(date_col):
        if len(g) < 3 or g[pred_col].nunique() < 3 or g[target].nunique() < 2:
            continue
        ndays += 1
        ics.append(safe_spearman(g[target].values, g[pred_col].values))
        k = min(topk, len(g))
        if k > 0:
            g_sorted = g.sort_values(pred_col, ascending=False).head(k)
            hits.append((g_sorted[target] > 0).mean())
    return {
        "IC_mean": float(np.nanmean(ics)) if ics else np.nan,
        "IC_median": float(np.nanmedian(ics)) if ics else np.nan,
        "HitRate@K": float(np.nanmean(hits)) if hits else np.nan,
        "days": int(ndays),
    }


def make_relevance_per_group(df: pd.DataFrame, group_col: str, target: str, n_bins: int = 5) -> pd.Series:
    """
    Integer-Relevanz 0..(n_bins-1) je Gruppe via Quantile.
    Nutzt index-alignte Series statt numpy-Array -> keine Out-of-bounds.
    """
    labels = pd.Series(index=df.index, dtype="float64")
    for _, g in df.groupby(group_col):
        if g[target].nunique() < 2 or len(g) < 3:
            labels.loc[g.index] = n_bins // 2
            continue
        try:
            q = pd.qcut(g[target], q=n_bins, labels=False, duplicates="drop")
            labels.loc[g.index] = q.values
        except Exception:
            labels.loc[g.index] = n_bins // 2
    return labels.fillna(n_bins // 2).astype(int).clip(lower=0, upper=n_bins - 1)



def prepare_features(
    df: pd.DataFrame,
    date_col: str,
    ticker_col: str,
    target: str,
    drop_extra: List[str],
    impute_value: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df[date_col] = safe_to_datetime(df[date_col])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])

    # Drop schlechte/konstante Spalten
    df = drop_constant_and_inf(df)

    # Feature-Liste
    drop_cols = [date_col, ticker_col] + drop_extra
    feats = infer_feature_cols(df, target=target, drop_cols=drop_cols)

    # Impute
    if feats:
        df[feats] = df[feats].fillna(impute_value)

    # Sortierung (wichtig für Ranking-Gruppen)
    df = df.sort_values([date_col, ticker_col]).reset_index(drop=True)
    return df, feats


# =========================
# I/O – Split Loader
# =========================
def iter_splits_from_parquets(parquet_dir: Path) -> List[int]:
    """
    Unterstützte Muster:
      - train_{i}.parquet  / test_{i}.parquet
      - scaled_train_{i}.parquet / scaled_test_{i}.parquet   (dein aktuelles Muster)
      - split_{iii}/train.parquet + split_{iii}/test.parquet
      - train.split{i}.parquet / test.split{i}.parquet
    """
    cand: set[int] = set()

    # Schema A
    for p in glob(str(parquet_dir / "train_*.parquet")):
        i = Path(p).stem.split("_")[-1]
        if (parquet_dir / f"test_{i}.parquet").exists():
            cand.add(int(i))

    # Schema B (scaled_*)
    for p in glob(str(parquet_dir / "scaled_train_*.parquet")):
        m = re.search(r"scaled_train_(\d+)\.parquet$", Path(p).name)
        if not m:
            continue
        i = int(m.group(1))
        if (parquet_dir / f"scaled_test_{i:03d}.parquet").exists():
            cand.add(i)

    # Schema C (Unterordner)
    for p in glob(str(parquet_dir / "split_*")):
        try:
            i = int(Path(p).name.replace("split_", ""))
        except Exception:
            continue
        if (Path(p) / "train.parquet").exists() and (Path(p) / "test.parquet").exists():
            cand.add(i)

    # Schema D
    for p in glob(str(parquet_dir / "train.split*.parquet")):
        i = Path(p).stem.split("split")[-1]
        if (parquet_dir / f"test.split{i}.parquet").exists():
            cand.add(int(i))

    return sorted(cand)


def load_split_parquets(parquet_dir: Path, split_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # A
    a = parquet_dir / f"train_{split_idx}.parquet"
    b = parquet_dir / f"test_{split_idx}.parquet"
    if a.exists() and b.exists():
        return pd.read_parquet(a), pd.read_parquet(b)
    # B (scaled_*)
    a = parquet_dir / f"scaled_train_{split_idx:03d}.parquet"
    b = parquet_dir / f"scaled_test_{split_idx:03d}.parquet"
    if a.exists() and b.exists():
        return pd.read_parquet(a), pd.read_parquet(b)
    # C (split_i/train.parquet)
    d = parquet_dir / f"split_{split_idx:03d}"
    if (d / "train.parquet").exists() and (d / "test.parquet").exists():
        return pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")
    # D (train.spliti.parquet)
    a = parquet_dir / f"train.split{split_idx}.parquet"
    b = parquet_dir / f"test.split{split_idx}.parquet"
    if a.exists() and b.exists():
        return pd.read_parquet(a), pd.read_parquet(b)

    raise FileNotFoundError(f"Keine Parquet-Paare für Split {split_idx} in {parquet_dir} gefunden.")


def load_split_from_csv(csv_path: Path, splits_log: Path, split_idx: int, set_name: str, date_col: str) -> pd.DataFrame:
    df_all = pd.read_csv(csv_path)
    if date_col in df_all.columns:
        df_all[date_col] = safe_to_datetime(df_all[date_col])

    sl = pd.read_csv(splits_log)
    if "split_id" not in sl.columns or "set" not in sl.columns or "start" not in sl.columns or "end" not in sl.columns:
        raise ValueError("splits_log.csv erwartet Spalten: split_id,set,start,end")

    s = sl[sl["split_id"] == split_idx]
    if set_name not in set(s["set"]):
        raise ValueError(f"Split {split_idx} enthält kein '{set_name}'")

    row = s[s["set"] == set_name].iloc[0]
    start = pd.to_datetime(row["start"])
    end = pd.to_datetime(row["end"])
    mask = (df_all[date_col] >= start) & (df_all[date_col] <= end)
    return df_all.loc[mask].copy()


# =========================
# Models
# =========================
def train_regressor(trn: pd.DataFrame, val: pd.DataFrame, feats: List[str], target: str, seed: int = 42) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=-1,
        objective="regression",
        verbose=-1,
    )
    model.fit(
        trn[feats],
        trn[target],
        eval_set=[(val[feats], val[target])],
        eval_metric="l2",
        callbacks=[],
    )
    return model


def group_lengths(df: pd.DataFrame, group_col: str) -> List[int]:
    return [len(g) for _, g in df.groupby(group_col)]


def train_ranker(
    trn: pd.DataFrame, val: pd.DataFrame, feats: List[str], label_col: str, group_col: str, seed: int = 42
) -> Optional[LGBMRanker]:
    g_trn = group_lengths(trn, group_col)
    g_val = group_lengths(val, group_col)
    if len(g_trn) < 5 or len(g_val) < 3:
        return None

    model = LGBMRanker(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=-1,
        objective="lambdarank",
        verbose=-1,
        # Optional könntest du die "Gewinne" je Klasse festlegen:
        # label_gain=[0,1,2,3,4],
    )
    model.fit(
        trn[feats],
        trn[label_col],
        group=g_trn,
        eval_set=[(val[feats], val[label_col])],
        eval_group=[g_val],
        eval_at=[5, 10],
    )
    return model


# =========================
# Evaluation pro Split
# =========================
def evaluate_split(
    split_idx: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
    ticker_col: str,
    target: str,
    group_col: str,
    topk: int,
    outdir: Path,
    do_ranker: bool,
    rel_bins: int,
) -> Dict[str, Dict[str, float]]:
    # Prepare
    drop_extra = [target]  # Target darf kein Feature sein
    trn_prep, feats = prepare_features(train_df, date_col, ticker_col, target, drop_extra, impute_value=0.0)
    tst_prep, _ = prepare_features(test_df, date_col, ticker_col, target, drop_extra, impute_value=0.0)

    # Align
    common_cols = sorted(set(trn_prep.columns) & set(tst_prep.columns))
    trn_prep = trn_prep[common_cols]
    tst_prep = tst_prep[common_cols]
    feats = [f for f in feats if f in common_cols]

    if not feats:
        log(f"[{split_idx:03d}] Keine gültigen Features nach Cleaning. Überspringe Split.")
        return {}

    # Val
    trn, val = make_val_split_by_time(trn_prep, date_col=date_col, val_frac=0.15)

    # -------- Regression --------
    reg = train_regressor(trn, val, feats, target)
    tst_prep["pred_reg"] = reg.predict(tst_prep[feats])
    reg_metrics = daily_metrics(tst_prep, date_col, target, "pred_reg", topk)

    fi_reg = pd.DataFrame({"feature": feats, "importance": reg.booster_.feature_importance()}).sort_values(
        "importance", ascending=False
    )

    # -------- Ranking (optional) --------
    rank_metrics = None
    if do_ranker:
        trn["rel_label"] = make_relevance_per_group(trn, group_col=group_col, target=target, n_bins=rel_bins)
        val["rel_label"] = make_relevance_per_group(val, group_col=group_col, target=target, n_bins=rel_bins)

        rnk = train_ranker(trn, val, feats, label_col="rel_label", group_col=group_col)
        if rnk is not None:
            tst_prep["pred_rank"] = rnk.predict(tst_prep[feats])
            rank_metrics = daily_metrics(tst_prep, date_col, target, "pred_rank", topk)
        else:
            log(f"[{split_idx:03d}] Ranker übersprungen (zu wenige Gruppen).")

    # -------- Persist --------
    split_dir = outdir / f"split_{split_idx:03d}"
    ensure_dir(split_dir)

    pred_cols = [date_col, ticker_col, target, "pred_reg"] + (["pred_rank"] if "pred_rank" in tst_prep.columns else [])
    tst_prep[pred_cols].to_csv(split_dir / "predictions.csv", index=False)
    fi_reg.to_csv(split_dir / "feature_importance_reg.csv", index=False)

    metrics: Dict[str, Dict[str, float]] = {"regression": reg_metrics}
    if rank_metrics is not None:
        metrics["lambda_rank"] = rank_metrics
    with open(split_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Log-Line
    reg_ic = metrics["regression"]["IC_mean"]
    reg_hit = metrics["regression"]["HitRate@K"]
    rnk_ic = metrics.get("lambda_rank", {}).get("IC_mean", np.nan)
    rnk_hit = metrics.get("lambda_rank", {}).get("HitRate@K", np.nan)
    log(
        f"[{split_idx:03d}] IC reg={reg_ic:.3f}, lambda={rnk_ic:.3f} | "
        f"hit reg={reg_hit:.3f}, lambda={rnk_hit:.3f} (n_days={metrics['regression']['days']})"
    )

    return metrics


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="QNM AMC – Modell-Evaluation über Rolling-Splits")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parquet-dir", type=str, help="Ordner mit train/test-Splits (mehrere Namensmuster unterstützt)")
    mode.add_argument("--csv", type=str, help="Pfad zur Gesamtdaten-CSV")
    ap.add_argument("--splits-log", type=str, help="Pfad zu splits_log.csv (nur mit --csv)")
    ap.add_argument("--date-col", type=str, default="Date")
    ap.add_argument("--ticker-col", type=str, default="Ticker")
    ap.add_argument("--target", type=str, default="Excess_5d_fwd")
    ap.add_argument("--group-col", type=str, default="Date", help="Ranking-Gruppierung (z. B. Date)")
    ap.add_argument("--topk", type=int, default=10, help="HitRate@K")
    ap.add_argument("--ranker", action="store_true", help="LambdaRank zusätzlich evaluieren (mit integer Relevanzlabels)")
    ap.add_argument("--rel-bins", type=int, default=5, help="Anzahl Klassen für Relevanzlabels (z. B. 5 -> 0..4)")
    ap.add_argument("--splits", type=str, default=None, help="Kommagetrennte Liste von Split-IDs (z. B. 0,1,2) – ansonsten auto")
    ap.add_argument("--outdir", type=str, default="reports/eval")
    args = ap.parse_args()

    date_col = args.date_col
    ticker_col = args.ticker_col
    target = args.target
    group_col = args.group_col
    topk = args.topk
    do_ranker = args.ranker
    rel_bins = args.rel_bins

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    summary_rows = []

    if args.parquet_dir:
        pq_dir = Path(args.parquet_dir)
        if args.splits:
            split_ids = [int(s) for s in args.splits.split(",")]
        else:
            split_ids = iter_splits_from_parquets(pq_dir)
        log(f"[INFO] Nutze Parquets. Splits gefunden: {len(split_ids)} -> {split_ids[:5]}{'...' if len(split_ids) > 5 else ''}")

        for i in split_ids:
            try:
                trn, tst = load_split_parquets(pq_dir, i)
                metrics = evaluate_split(
                    split_idx=i,
                    train_df=trn,
                    test_df=tst,
                    date_col=date_col,
                    ticker_col=ticker_col,
                    target=target,
                    group_col=group_col,
                    topk=topk,
                    outdir=outdir,
                    do_ranker=do_ranker,
                    rel_bins=rel_bins,
                )
                if metrics:
                    row = {
                        "split": i,
                        "reg_IC_mean": metrics["regression"]["IC_mean"],
                        "reg_IC_median": metrics["regression"]["IC_median"],
                        "reg_HitRate@K": metrics["regression"]["HitRate@K"],
                        "reg_days": metrics["regression"]["days"],
                        "rank_IC_mean": metrics.get("lambda_rank", {}).get("IC_mean", np.nan),
                        "rank_IC_median": metrics.get("lambda_rank", {}).get("IC_median", np.nan),
                        "rank_HitRate@K": metrics.get("lambda_rank", {}).get("HitRate@K", np.nan),
                        "rank_days": metrics.get("lambda_rank", {}).get("days", np.nan),
                    }
                    summary_rows.append(row)
            except FileNotFoundError as e:
                log(f"[{i:03d}] Dateien nicht gefunden – Split übersprungen. ({e})")
            except Exception as e:
                log(f"[{i:03d}] Fehler: {e}")

    else:
        # CSV + splits_log
        if not args.splits_log:
            raise ValueError("--splits-log ist erforderlich mit --csv")
        csv_path = Path(args.csv)
        sl_path = Path(args.splits_log)

        if args.splits:
            split_ids = [int(s) for s in args.splits.split(",")]
        else:
            sl = pd.read_csv(sl_path)
            if "split_id" not in sl.columns:
                raise ValueError("splits_log.csv benötigt Spalte 'split_id'")
            split_ids = sorted(sl["split_id"].unique().tolist())

        log(f"[INFO] Nutze CSV + splits_log. Splits: {len(split_ids)}")

        for i in split_ids:
            try:
                trn = load_split_from_csv(csv_path, sl_path, i, "train", date_col)
                tst = load_split_from_csv(csv_path, sl_path, i, "test", date_col)
                metrics = evaluate_split(
                    split_idx=i,
                    train_df=trn,
                    test_df=tst,
                    date_col=date_col,
                    ticker_col=ticker_col,
                    target=target,
                    group_col=group_col,
                    topk=topk,
                    outdir=outdir,
                    do_ranker=do_ranker,
                    rel_bins=rel_bins,
                )
                if metrics:
                    row = {
                        "split": i,
                        "reg_IC_mean": metrics["regression"]["IC_mean"],
                        "reg_IC_median": metrics["regression"]["IC_median"],
                        "reg_HitRate@K": metrics["regression"]["HitRate@K"],
                        "reg_days": metrics["regression"]["days"],
                        "rank_IC_mean": metrics.get("lambda_rank", {}).get("IC_mean", np.nan),
                        "rank_IC_median": metrics.get("lambda_rank", {}).get("IC_median", np.nan),
                        "rank_HitRate@K": metrics.get("lambda_rank", {}).get("HitRate@K", np.nan),
                        "rank_days": metrics.get("lambda_rank", {}).get("days", np.nan),
                    }
                    summary_rows.append(row)
            except Exception as e:
                log(f"[{i:03d}] Fehler: {e}")

    # Gesamtsummary speichern
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values("split")
        df_sum.to_csv(outdir / "summary_metrics.csv", index=False)
        log(f"[INFO] Summary gespeichert: {outdir / 'summary_metrics.csv'}")
        # Kurz-Überblick
        mean_ic = float(np.nanmean(df_sum["reg_IC_mean"]))
        mean_hit = float(np.nanmean(df_sum["reg_HitRate@K"]))
        log(f"[INFO] Ø über Splits – IC(reg)={mean_ic:.3f}, HitRate@K(reg)={mean_hit:.3f}")

    log("[INFO] Fertig.")


if __name__ == "__main__":
    main()
