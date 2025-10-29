import os, json
import numpy as np
import pandas as pd
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Sanity-Checks für scaled Parquets")
    p.add_argument("--scaled-dir", default=r"C:\Users\holze\PycharmProjects\QNM-AMC\reports\scaled")
    p.add_argument("--reports-dir", default=r"C:\Users\holze\PycharmProjects\QNM-AMC\reports")
    p.add_argument("--split-id", type=int, default=0, help="Welcher Split (default 0)")
    p.add_argument("--show-corr", action="store_true", help="Kleine Korrelationstabelle printen")
    return p.parse_args()

def main():
    a = parse_args()
    sid = a.split_id

    # Dateien
    train_pq = os.path.join(a.scaled_dir, f"scaled_train_{sid:03d}.parquet")
    test_pq  = os.path.join(a.scaled_dir, f"scaled_test_{sid:03d}.parquet")
    meta_js  = os.path.join(a.scaled_dir, f"meta_{sid:03d}.json")
    idx_prefix = "splits_log_idx"  # passt zu deinem prepare_splits
    rid_tr = os.path.join(a.reports_dir, f"{idx_prefix}_{sid:03d}_train_rid.npy")
    rid_te = os.path.join(a.reports_dir, f"{idx_prefix}_{sid:03d}_test_rid.npy")

    # Laden
    train = pd.read_parquet(train_pq)
    test  = pd.read_parquet(test_pq)
    meta  = json.load(open(meta_js, "r", encoding="utf-8"))

    print(f"[SPLIT {sid:03d}] shapes train/test:", train.shape, test.shape)
    print("[META] keys:", sorted(meta.keys()))

    # NaN-Quote der ersten paar Spalten
    print("\n[NaN-Anteil (Top 10 Spalten)]")
    print(train.isna().mean().sort_values(ascending=False).head(10))

    # Featureliste (ohne Schlüssel/Ziel anpassen wenn dein Target anders heißt)
    target = meta.get("target", "Excess_5d_fwd")
    skip = {"Date", "Ticker", target}
    feat = [c for c in train.columns if c not in skip and pd.api.types.is_numeric_dtype(train[c])]

    # Grund-Stats (mean/std) auf ein paar Features
    few = feat[:10] if len(feat) >= 10 else feat
    desc = train[few].describe().T.loc[:, ["mean", "std"]]
    print("\n[Mean/Std (erste Features)]")
    print(desc)

    # Pro-Ticker-Check (zeigt, dass Z-Score je Ticker ~0/1 ist)
    cols_for_group = [c for c in ["Price","Volume","MktCap"] if c in train.columns]
    if cols_for_group:
        print("\n[Pro Ticker mean/std (Sample)]")
        print(train.groupby("Ticker")[cols_for_group].agg(["mean","std"]).head(5))

    # Zielvariable
    if target in train.columns:
        print(f"\n[Target '{target}' describe()]")
        print(train[target].describe())

    # Korrelation (optional)
    if a.show_corr:
        cand = [c for c in ["Return_raw","Return_lag1","Return_lag5"] if c in train.columns]
        if len(cand) >= 2:
            print("\n[Korrelation ausgewählter Returns]")
            print(train[cand].corr())

    # RID-Guards (Konsistenz)
    if os.path.exists(rid_tr) and os.path.exists(rid_te):
        rid_train = np.load(rid_tr)
        rid_test  = np.load(rid_te)
        print("\n[RID-Guards]")
        print("len(rid_train), len(rid_test):", len(rid_train), len(rid_test))
    else:
        print("\n[RID-Guards] Dateien nicht gefunden – ok, wenn du sie nicht nutzt.")

if __name__ == "__main__":
    main()
