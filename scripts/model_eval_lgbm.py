from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit("LightGBM wird benötigt: pip install lightgbm")


# ----------------------------- CLI & Setup ----------------------------- #

def setup_logging(verbose: bool):
    logging.basicConfig(level=(logging.DEBUG if verbose else logging.INFO),
                        format='[%(levelname)s] %(message)s')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet-dir', required=True, type=Path)
    p.add_argument('--target', default='Excess_5d_fwd')
    p.add_argument('--ranker', action='store_true', help='LambdaRank statt Regressor')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--outdir', required=True, type=Path)
    p.add_argument('--clip-quantiles', nargs=2, type=float, default=None,
                   help='zwei Quantile fürs Winsorizing, z.B. 0.005 0.995')
    p.add_argument('--loss', choices=['huber','mse'], default='huber', help='nur Regressor')
    p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--num-leaves', type=int, default=31)
    p.add_argument('--n-estimators', type=int, default=500)
    p.add_argument('--early-stopping', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cost-bps', type=float, default=10.0)
    p.add_argument('--max-weight-per-name', type=float, default=0.10)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


# ----------------------------- Data structures ----------------------------- #

@dataclass
class SplitData:
    split_id: int
    train: pd.DataFrame
    test: pd.DataFrame


# ----------------------------- File discovery ----------------------------- #

def find_splits(pdir: Path) -> list[int]:
    """Findet Split-IDs für mehrere Namensmuster (case-insensitive):
    - train_000.parquet / test_000.parquet
    - scaled_train_000.parquet / scaled_test_000.parquet
    - split_000_train.parquet / split_000_test.parquet
    - 000_train.parquet / 000_test.parquet
    """
    import re
    ids: set[int] = set()
    for f in pdir.glob('*.parquet'):
        name = f.name.lower()
        for rx in [
            r'^train_(\d+)\.parquet$', r'^test_(\d+)\.parquet$',
            r'^scaled_train_(\d+)\.parquet$', r'^scaled_test_(\d+)\.parquet$',
            r'^split_(\d+)_train\.parquet$', r'^split_(\d+)_test\.parquet$',
            r'^(\d+)_train\.parquet$', r'^(\d+)_test\.parquet$']:
            m = re.match(rx, name)
            if m:
                ids.add(int(m.group(1)))
                break
    return sorted(ids)


def load_split(pdir: Path, sid: int) -> SplitData:
    patterns = [
        (f'train_{sid:03d}.parquet', f'test_{sid:03d}.parquet'),
        (f'scaled_train_{sid:03d}.parquet', f'scaled_test_{sid:03d}.parquet'),
        (f'split_{sid:03d}_train.parquet', f'split_{sid:03d}_test.parquet'),
        (f'{sid:03d}_train.parquet', f'{sid:03d}_test.parquet'),
    ]
    tpath = vpath = None
    for tpat, vpat in patterns:
        t, v = pdir / tpat, pdir / vpat
        if t.exists() and v.exists():
            tpath, vpath = t, v
            break
    if tpath is None:
        listing = ', '.join(sorted(p.name for p in pdir.glob('*.parquet'))) or '(keine)'
        raise FileNotFoundError(f'Keine Parquets für Split {sid:03d}. Gefunden: {listing}')

    train = pd.read_parquet(tpath)
    test = pd.read_parquet(vpath)
    for df in (train, test):
        if 'Date' in df.columns and not np.issubdtype(df['Date'].dtype, np.datetime64):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return SplitData(split_id=sid, train=train, test=test)


# ----------------------------- Preprocessing ----------------------------- #

def winsorize_target(df: pd.DataFrame, target: str, q_lo: float, q_hi: float) -> pd.Series:
    lo, hi = df[target].quantile([q_lo, q_hi])
    return df[target].clip(lower=lo, upper=hi)


def get_features(df: pd.DataFrame, target: str) -> list[str]:
    cols = [c for c in df.columns if c not in ('Date','Ticker', target)]
    feats = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    return feats


def group_lengths_by_date(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_values(['Date'])
    return df.groupby('Date').size().values


def make_relevance(y: pd.Series, dates: pd.Series, method: str = 'rank', bins: int = 5) -> np.ndarray:
    """Mappe kontinuierliche Targets je Datum auf int-Relevanzen ≥ 0.
    - method='rank': dichte Ränge 0..(n-1)
    - method='quantile': Quantil-Bins 0..(bins-1)
    """
    y = pd.Series(y).reset_index(drop=True)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    frame = pd.DataFrame({'y': y, 'Date': dates})
    rel = np.zeros(len(frame), dtype=int)
    for d, g in frame.groupby('Date', sort=False):
        if len(g) <= 1:
            continue
        if method == 'quantile' and len(g) >= bins:
            q = pd.qcut(g['y'].rank(method='first'), q=bins, labels=False, duplicates='drop')
            rel[g.index] = q.astype(int)
        else:
            r = g['y'].rank(method='first').astype(int) - 1
            rel[g.index] = r.values
    return rel


# ----------------------------- Metrics & Backtest ----------------------------- #

def ic_by_day(df: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby('Date'):
        s = g[score_col]
        y = g[target_col]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if s.nunique() <= 1:
            continue
        y_std = float(np.nanstd(y))
        if y_std == 0.0 or np.isnan(y_std):
            continue
        sp = spearmanr(s, y, nan_policy='omit')[0]
        try:
            pe = pearsonr(s, y)[0]
        except Exception:
            pe = np.nan
        rows.append({'Date': d, 'IC_spearman': sp, 'IC_pearson': pe, 'n': len(g)})
    out = pd.DataFrame(rows)
    return out.sort_values('Date') if not out.empty else out


def topk_hit_rate(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> float:
    hits = []
    for _, g in df.groupby('Date'):
        g = g.sort_values(score_col, ascending=False)
        k_eff = min(k, len(g))
        if k_eff == 0:
            continue
        top = g.head(k_eff)
        thresh = g[target_col].median()
        hits.append(float((top[target_col] > thresh).mean()))
    return float(np.mean(hits)) if hits else np.nan


def backtest_topk(df: pd.DataFrame, score_col: str, ret_col: str, k: int,
                  cost_bps: float, max_w: float) -> pd.DataFrame:
    daily = []
    prev_weights = None
    bps = cost_bps / 1e4  # 10 bps = 0.001
    for d, g in df.groupby('Date'):
        g = g.sort_values(score_col, ascending=False)
        k_eff = min(k, len(g))
        if k_eff == 0:
            continue
        picks = g.head(k_eff)[['Ticker', ret_col]].copy()
        w = np.full(k_eff, 1.0 / k_eff)
        w = np.minimum(w, max_w)
        w = w / w.sum()
        picks['w'] = w

        if prev_weights is None:
            turnover = float(np.sum(np.abs(w)))
        else:
            merged = pd.merge(picks[['Ticker','w']], prev_weights[['Ticker','w']],
                              on='Ticker', how='outer', suffixes=('', '_prev')).fillna(0.0)
            turnover = float(0.5 * np.sum(np.abs(merged['w'] - merged['w_prev'])))

        gross = float(np.sum(picks[ret_col].values * w))
        net = gross - turnover * bps
        daily.append({'Date': d, 'gross': gross, 'net': net, 'turnover': turnover})
        prev_weights = picks[['Ticker','w']]
    return pd.DataFrame(daily).sort_values('Date')


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()) if len(dd) else np.nan


# ----------------------------- Models ----------------------------- #

def fit_predict_regressor(Xtr, ytr, Xte, params, seed):
    model = lgb.LGBMRegressor(
        objective=('huber' if params['loss']=='huber' else 'mse'),
        learning_rate=params['lr'], num_leaves=params['leaves'],
        n_estimators=params['estimators'], random_state=seed,
        subsample=0.9, colsample_bytree=0.9,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, np.zeros(len(Xte)))], verbose=False)
    return model, model.predict(Xtr), model.predict(Xte)


def fit_predict_ranker(Xtr, ytr, dates_tr, Xte, dates_te, group_tr, group_te, params, seed):
    # int-Relevanzen je Datum (dichte Ränge)
    rel_tr = make_relevance(ytr, dates_tr, method='rank')
    max_rel = int(rel_tr.max()) if len(rel_tr) else 0
    label_gain = list(range(max_rel + 1))

    lgb_train = lgb.Dataset(Xtr, label=rel_tr, group=group_tr)
    # Test-Labels: int Pflicht, Inhalt egal
    rel_te = np.zeros(len(Xte), dtype=int)
    lgb_test = lgb.Dataset(Xte, label=rel_te, group=group_te, reference=lgb_train)

    cfg = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'learning_rate': params['lr'], 'num_leaves': params['leaves'],
        'n_estimators': params['estimators'], 'verbosity': -1,
        'seed': seed, 'feature_pre_filter': False, 'label_gain': label_gain,
    }
    model = lgb.train(cfg, lgb_train, num_boost_round=params['estimators'], valid_sets=[lgb_test])
    return model, model.predict(Xtr), model.predict(Xte)


# ----------------------------- Main ----------------------------- #

def main():
    args = parse_args()
    setup_logging(args.verbose)
    args.outdir.mkdir(parents=True, exist_ok=True)

    split_ids = find_splits(args.parquet_dir)
    if not split_ids:
        logging.error('Keine Splits gefunden.')
        return
    logging.info(f'Nutze Parquets. Splits gefunden: {len(split_ids)} -> {split_ids[:5]}...')

    rng = np.random.default_rng(args.seed)

    summary_rows = []
    eq_all = []

    for sid in split_ids:
        sd = load_split(args.parquet_dir, sid)
        train, test = sd.train.copy(), sd.test.copy()

        # Winsorizing (Quantile auf TRAIN bestimmen)
        if args.clip_quantiles is not None:
            q_lo, q_hi = args.clip_quantiles
            train[args.target] = winsorize_target(train, args.target, q_lo, q_hi)
            lo, hi = train[args.target].quantile([q_lo, q_hi])
            test[args.target] = test[args.target].clip(lower=lo, upper=hi)
            logging.debug(f'[{sid:03d}] Target winsorized @ {q_lo}/{q_hi}')

        feats = [f for f in get_features(train, args.target) if f in test.columns]
        # Sortierung für konsistente Gruppierung
        train = train.sort_values(['Date'])
        test = test.sort_values(['Date'])
        Xtr, ytr = train[feats], train[args.target]
        Xte, yte = test[feats], test[args.target]
        group_tr = group_lengths_by_date(train)
        group_te = group_lengths_by_date(test)
        n_days = int(len(group_te))
        if n_days == 0:
            logging.warning(f'[{sid:03d}] n_days=0 → Split übersprungen')
            continue

        params = {
            'loss': args.loss, 'lr': args.learning_rate,
            'leaves': args.num_leaves, 'estimators': args.n_estimators,
        }

        if args.ranker:
            model, pred_tr, pred_te = fit_predict_ranker(Xtr, ytr.values, train['Date'].values,
                                                         Xte, test['Date'].values,
                                                         group_tr, group_te, params, args.seed)
        else:
            model, pred_tr, pred_te = fit_predict_regressor(Xtr, ytr.values, Xte, params, args.seed)

        # Predictions & Eval-Frames
        test_out = test[['Date','Ticker',args.target]].copy()
        test_out['y_pred'] = pred_te

        # Metriken
        ic_df = ic_by_day(test_out, 'y_pred', args.target)
        ic_mean = float(ic_df['IC_spearman'].mean()) if not ic_df.empty else np.nan
        ic_std = float(ic_df['IC_spearman'].std(ddof=0)) if not ic_df.empty else np.nan
        hit = topk_hit_rate(test_out, 'y_pred', args.target, args.topk)

        bt = backtest_topk(test_out, 'y_pred', args.target, args.topk, args.cost_bps, args.max_weight_per_name)
        if bt.empty:
            logging.warning(f'[{sid:03d}] Backtest leer.')
            continue
        eq = (1.0 + bt['net']).cumprod()
        cagr = float(eq.iloc[-1] ** (252/len(eq)) - 1.0) if len(eq) else np.nan
        vol = float(np.std(bt['net']) * np.sqrt(252)) if len(bt) else np.nan
        sharpe = float(np.mean(bt['net']) / np.std(bt['net']) * np.sqrt(252)) if bt['net'].std(ddof=0) > 0 else np.nan
        maxdd = max_drawdown(eq)
        avg_turn = float(bt['turnover'].mean()) if len(bt) else np.nan

        logging.info(f'[{sid:03d}] IC={ic_mean:.3f} | hit={hit:.3f} | n_days={n_days}')

        # Dumps pro Split
        sdir = args.outdir / f'split_{sid:03d}'
        sdir.mkdir(parents=True, exist_ok=True)
        test_out.to_csv(sdir / 'predictions_test.csv', index=False)
        ic_df.to_csv(sdir / 'ic_by_day.csv', index=False)
        bt.to_csv(sdir / 'backtest_daily.csv', index=False)

        # Placebo (Shuffle Target je Tag) – eigene Zielspalte, kein Rename
        placebo = test_out.copy()
        placebo['y_true_shuffled'] = placebo.groupby('Date')[args.target].transform(lambda x: rng.permutation(x.values))
        ic_placebo = ic_by_day(placebo, 'y_pred', 'y_true_shuffled')
        ic_placebo_mean = float(ic_placebo['IC_spearman'].mean()) if not ic_placebo.empty else np.nan

        # Shift-Check: Target -1 Tag je Ticker → neue Spalte
        shifted = test_out.copy()
        shifted['y_shiftm1'] = (shifted.sort_values(['Ticker','Date'])
                                 .groupby('Ticker')[args.target].shift(-1))
        shifted = shifted.dropna(subset=['y_shiftm1'])
        ic_shift = ic_by_day(shifted, 'y_pred', 'y_shiftm1')
        ic_shift_mean = float(ic_shift['IC_spearman'].mean()) if not ic_shift.empty else np.nan

        # Plots pro Split
        plt.figure()
        plt.plot(bt['Date'], (1.0 + bt['net']).cumprod(), label='Portfolio (net)')
        plt.title(f'Equity Curve – Split {sid:03d}')
        plt.xlabel('Date'); plt.ylabel('Equity'); plt.legend(); plt.tight_layout()
        plt.savefig(sdir / 'equity_curve.png'); plt.close()

        plt.figure()
        plt.hist(ic_df['IC_spearman'].dropna(), bins=30)
        plt.title(f'IC (Spearman) – Split {sid:03d}')
        plt.xlabel('IC'); plt.ylabel('Häufigkeit'); plt.tight_layout()
        plt.savefig(sdir / 'ic_hist.png'); plt.close()

        # Summary sammeln
        summary_rows.append({
            'split_id': sid,
            'IC_mean': ic_mean, 'IC_std': ic_std,
            'HitRate_topk': hit,
            'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': maxdd,
            'AvgTurnover': avg_turn,
            'IC_placebo_mean': ic_placebo_mean, 'IC_shiftm1_mean': ic_shift_mean,
            'n_days': n_days,
        })

        eq_part = bt[['Date','net']].copy(); eq_part['split_id'] = sid; eq_all.append(eq_part)

    if not summary_rows:
        logging.error('Keine validen Splits ausgewertet.')
        return

    summary = pd.DataFrame(summary_rows).sort_values('split_id')
    summary.to_csv(args.outdir / 'eval_summary.csv', index=False)

    # Aggregatplots
    plt.figure(); plt.bar(summary['split_id'].astype(str), summary['IC_mean'].values)
    plt.title('IC (Spearman) je Split'); plt.xlabel('Split'); plt.ylabel('IC_mean'); plt.tight_layout()
    plt.savefig(args.outdir / 'ic_by_split.png'); plt.close()

    plt.figure(); plt.bar(summary['split_id'].astype(str), summary['HitRate_topk'].values)
    plt.title('Top-k Hit-Rate je Split'); plt.xlabel('Split'); plt.ylabel('Hit-Rate'); plt.tight_layout()
    plt.savefig(args.outdir / 'hit_rate_by_split.png'); plt.close()

    plt.figure(); plt.bar(summary['split_id'].astype(str), summary['AvgTurnover'].values)
    plt.title('Durchschnittlicher Turnover je Split'); plt.xlabel('Split'); plt.ylabel('Avg Turnover'); plt.tight_layout()
    plt.savefig(args.outdir / 'turnover_by_split.png'); plt.close()

    # Gesamt-Equity (vereinfacht verkettet)
    eq_all_df = pd.concat(eq_all, ignore_index=True).sort_values('Date')
    eq_curve = (1.0 + eq_all_df['net']).cumprod()
    plt.figure(); plt.plot(eq_all_df['Date'], eq_curve)
    plt.title('Gesamt-Equity (verkettet)'); plt.xlabel('Date'); plt.ylabel('Equity (net)'); plt.tight_layout()
    plt.savefig(args.outdir / 'equity_curve_all.png'); plt.close()

    # Konsolen-Zusammenfassung
    cons = {
        'IC_mean': float(summary['IC_mean'].mean()),
        'HitRate_topk': float(summary['HitRate_topk'].mean()),
        'CAGR_median': float(summary['CAGR'].median()),
        'Sharpe_median': float(summary['Sharpe'].median()),
        'MaxDD_median': float(summary['MaxDD'].median()),
        'AvgTurnover_mean': float(summary['AvgTurnover'].mean()),
        'IC_placebo_mean': float(summary['IC_placebo_mean'].mean()),
        'IC_shiftm1_mean': float(summary['IC_shiftm1_mean'].mean()),
        'n_splits': int(len(summary)),
    }
    print(json.dumps(cons, indent=2, default=float))

    with open(args.outdir / 'run_config.json', 'w', encoding='utf-8') as f:
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump({'args': args_dict, 'notes': 'neu & robust'}, f, indent=2)


if __name__ == '__main__':
    main()
