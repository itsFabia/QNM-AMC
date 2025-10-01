# stock_ml_baseline.py
import warnings; warnings.filterwarnings("ignore")

# ------------------------------------------------
# Imports
# ------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Eigene Module
from top_performers import compute_top_outperformers
from top_performers_SMI import SMI_TICKERS, SMI_BENCH  # ^SSMI

# ------------------------------------------------
# Globale Einstellungen
# ------------------------------------------------
MACRO_TICKERS = {
    "vix": "^VIX",
    "rates": "^TNX",  # US 10Y als Proxy für globale Zinserwartungen
    "eurchf": "EURCHF=X",
}

# ------------------------------------------------
# Globale Einstellungen
# ------------------------------------------------
LOOKBACK_YEARS = 20
TOP_N = 5
DATA_START = "2010-01-01"   # großzügig; Lookback schneidet das Fenster
PRED_THRESH = 0.6
FEE_BPS = 2
MIN_HOLD_DAYS = 20
N_SPLITS = 5
ROLL_BETA = 60  # für Beta/Excess-Return
WALK_TEST_WINDOW = 63  # ca. 3 Monate Handelstage
WALK_MIN_TRAIN = 252 * 2  # Zwei Jahre Startfenster

# ------------------------------------------------
# 1) Top-5 Outperformer vs. SMI (10 Jahre)
# ------------------------------------------------
top5, smi_tr = compute_top_outperformers(
    index_members=SMI_TICKERS,
    benchmark_ticker=SMI_BENCH,
    start=DATA_START,
    lookback_years=LOOKBACK_YEARS,
    top_n=TOP_N,
)
print("\nTop 5 Outperformer vs. SMI (10 Jahre):")
print(top5)

# ------------------------------------------------
# Hilfsfunktionen: Spalten-Picking & TR-Loader (robust)
# ------------------------------------------------
def _pick(colnames, candidates):
    cols = [str(c).lower().replace(" ", "_") for c in colnames]
    # exakte Treffer
    for cand in candidates:
        if not cand: continue
        c = cand.lower()
        if c in cols:
            return cols[cols.index(c)]
    # endet-mit (für Suffixe wie close_nesn.sw)
    for cand in candidates:
        if not cand: continue
        c = cand.lower()
        for col in cols:
            if col.endswith(c):
                return col
    return None

def _flatten_cols(columns):
    # MultiIndex -> Strings, ansonsten Stringify
    try:
        import pandas as pd  # local scope
        if isinstance(columns, pd.MultiIndex):
            return [
                "_".join([str(x) for x in col if x is not None and str(x) != ""])
                for col in columns.values
            ]
        return [str(c) for c in columns]
    except Exception:
        return [str(c) for c in columns]

def _load_tr_series(ticker: str, start: str) -> pd.Series:
    d = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if d.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    d.columns = _flatten_cols(d.columns)
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    t = ticker.lower()
    col = _pick(d.columns, ["adj_close", "adjclose", "close",
                            f"adj_close_{t}", f"adjclose_{t}", f"close_{t}"])
    if col is None:
        raise KeyError(f"{ticker}: keine Adj/Close-Spalte gefunden in {list(d.columns)}")
    px = d[col]
    ret = px.pct_change().fillna(0.0)
    return (1 + ret).cumprod()

# ------------------------------------------------
# Plot: SMI (TR) + Top-5 (normiert)
# ------------------------------------------------
plt.figure(figsize=(11,6))
smi_tr.plot(label="SMI (TR)")
for t in top5.index:
    try:
        tr = _load_tr_series(t, DATA_START)
        merged = pd.concat([smi_tr.rename("SMI"), tr.rename(t)], axis=1).dropna()
        if len(merged) < 2:
            print(f"Zu wenig überlappende Daten für {t} – überspringe.")
            continue
        (merged[t] / merged[t].iloc[0]).plot(label=t)
    except Exception as e:
        print(f"Plot-Skip {t}: {e}")
plt.title("Top-5 vs. SMI – Total Return (normiert)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 2) ML-Baseline: Outperformance ggü. Index (für Top-1-Ticker)
# ------------------------------------------------
TICKER = top5.index[0]
print(f"\nML-Baseline auf Outperformance vs. Index für: {TICKER}")

def _normalize_ohlc(df, ticker=None):
    """Normiert Yahoo-Download auf: adj_close, close, open, high, low, volume."""
    df.columns = _flatten_cols(df.columns)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    t = (ticker or "").lower()
    out = {}

    # adj_close / close
    adjc = _pick(df.columns, ["adj_close", "adjclose", f"adj_close_{t}", f"adjclose_{t}"])
    clos = _pick(df.columns, ["close", f"close_{t}"])
    if adjc is not None:
        out["adj_close"] = df[adjc]
    if clos is not None:
        out["close"] = df[clos]

    # open / high / low / volume
    for k in ["open", "high", "low", "volume"]:
        col = _pick(df.columns, [k, f"{k}_{t}"])
        if col is not None:
            out[k] = df[col]

    if not out.get("adj_close") and not out.get("close"):
        raise KeyError(f"{ticker or 'Asset'}: keine Adj/Close-Spalte gefunden in {list(df.columns)}")

    return pd.DataFrame(out)


def _load_macro_series(start: str) -> pd.DataFrame:
    macro = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if raw.empty:
                print(f"Warnung: keine Makrodaten für {ticker}")
                continue
            norm = _normalize_ohlc(raw, ticker)
            series = norm["adj_close"] if "adj_close" in norm.columns else norm.iloc[:, 0]
            macro[name] = series.rename(name)
        except Exception as exc:
            print(f"Makro-Skip {ticker}: {exc}")
    if not macro:
        return pd.DataFrame()
    return pd.concat(macro.values(), axis=1)


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "macd_signal": signal_line, "macd_hist": hist})


def _compute_bbands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / (ma + 1e-12)
    return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower, "bb_width": width})

# Preise laden (Asset + Benchmark)
asset_raw = yf.download(TICKER, start=DATA_START, auto_adjust=True, progress=False)
bench_raw = yf.download(SMI_BENCH, start=DATA_START, auto_adjust=True, progress=False)
if asset_raw.empty: raise ValueError(f"Keine Daten für {TICKER}")
if bench_raw.empty: raise ValueError(f"Keine Daten für {SMI_BENCH}")

a = _normalize_ohlc(asset_raw, TICKER)
b = _normalize_ohlc(bench_raw, SMI_BENCH)

# Close-Serien (Adj bevorzugt)
a_close = (a["adj_close"] if "adj_close" in a.columns else a["close"]).rename("a_close")
b_close = (b["adj_close"] if "adj_close" in b.columns else b["close"]).rename("b_close")

# Gemeinsames DF
df = pd.concat([a_close, b_close], axis=1).dropna()
df["ret"] = df["a_close"].pct_change()
df["bm_ret"] = df["b_close"].pct_change()

# Rolling-Beta & Excess-Return (beta-gehedged)
cov = df["ret"].rolling(ROLL_BETA).cov(df["bm_ret"])
var = df["bm_ret"].rolling(ROLL_BETA).var() + 1e-12
df["beta"] = (cov / var).clip(-5, 5)
df["excess_ret"] = df["ret"] - df["beta"] * df["bm_ret"]

# Ziel: Outperformance morgen
df["target"] = (df["excess_ret"].shift(-1) > 0).astype(int)

# ------------------------------------------------
# Features (markt-relativ) – nutzt normalisierte a/b
# ------------------------------------------------
def make_features_market_rel(d, asset_df, macro=None, windows=(3, 5, 10, 20)):
    out = d.copy()
    close = out["a_close"]
    bench_close = out["b_close"]
    excess = out["excess_ret"]
    ret = out["ret"]
    bm_ret = out["bm_ret"]

    for w in windows:
        out[f"ret_{w}d"] = close.pct_change(w)
        out[f"bm_ret_{w}d"] = bench_close.pct_change(w)
        out[f"ret_ex_{w}d"] = out[f"ret_{w}d"] - out[f"bm_ret_{w}d"]
        out[f"vol_{w}d"] = ret.rolling(w).std()
        out[f"bm_vol_{w}d"] = bm_ret.rolling(w).std()
        rolling_excess = excess.rolling(w)
        z_num = excess - rolling_excess.mean()
        z_den = rolling_excess.std() + 1e-9
        out[f"zex_{w}d"] = z_num / z_den
        out[f"beta_{w}d"] = out["beta"].rolling(w).mean()
        if "high" in asset_df.columns and "low" in asset_df.columns:
            hi = asset_df["high"].rolling(w).max()
            lo = asset_df["low"].rolling(w).min()
            out[f"hi_lo_{w}d"] = (hi - lo) / close
        if "volume" in asset_df.columns:
            out[f"volchg_{w}d"] = asset_df["volume"].pct_change(w)

    momentum_windows = {"1m": 21, "3m": 63, "6m": 126}
    for label, window in momentum_windows.items():
        out[f"momentum_{label}"] = close.pct_change(window)

    rel_strength = close / (bench_close + 1e-12)
    out["rel_strength"] = rel_strength
    out["rel_strength_trend"] = rel_strength.pct_change(63)

    out["rsi_14"] = _compute_rsi(close)
    macd = _compute_macd(close)
    out = out.join(macd)
    out["macd_cross"] = macd["macd"] - macd["macd_signal"]
    ma_fast = close.rolling(10).mean()
    ma_slow = close.rolling(50).mean()
    out["ma_10_50_diff"] = ma_fast - ma_slow
    out["ma_20_100_ratio"] = close.rolling(20).mean() / (close.rolling(100).mean() + 1e-12) - 1
    bb = _compute_bbands(close)
    out = out.join(bb)
    out["bb_pos"] = (close - bb["bb_lower"]) / (bb["bb_upper"] - bb["bb_lower"] + 1e-12)

    if "volume" in asset_df.columns:
        volume = asset_df["volume"]
        mean_20 = volume.rolling(20).mean()
        std_20 = volume.rolling(20).std()
        out["volume_z_20"] = (volume - mean_20) / (std_20 + 1e-12)
        out["volume_rel_20_60"] = mean_20 / (volume.rolling(60).mean() + 1e-12)

    if macro is not None and not macro.empty:
        macro = macro.reindex(out.index).ffill()
        for col in macro.columns:
            out[f"{col}_level"] = macro[col]
            out[f"{col}_ret_5d"] = macro[col].pct_change(5)
            z_mean = macro[col].rolling(20).mean()
            z_std = macro[col].rolling(20).std() + 1e-12
            out[f"{col}_z_20"] = (macro[col] - z_mean) / z_std

    out["dow"] = out.index.dayofweek
    out = pd.get_dummies(out, columns=["dow"], drop_first=True)
    out = out.dropna()
    return out

macro_df = _load_macro_series(DATA_START)
df_feat = make_features_market_rel(df, a, macro=macro_df)

drop_cols = ["target","a_close","b_close","ret","bm_ret","beta","excess_ret"]
feature_cols = [c for c in df_feat.columns if c not in drop_cols]
X = df_feat[feature_cols]
y = df_feat["target"]
dates = df_feat.index

pos_weight = float((len(y) - y.sum()) / (y.sum() + 1e-9))

# ------------------------------------------------
# Modelle & Cross-Validation
# ------------------------------------------------
logit = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

xgb_base_params = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    use_label_encoder=False,
)


def make_xgb_classifier(**kwargs):
    params = xgb_base_params.copy()
    params.update(kwargs)
    return XGBClassifier(**params)


rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced_subsample",
)


def tune_xgb_hyperparams(X, y, base_weight, param_grid, n_splits=3):
    best_auc = -np.inf
    best_params = {}
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for params in ParameterGrid(param_grid):
        aucs = []
        for tr, te in splitter.split(X):
            model = make_xgb_classifier(scale_pos_weight=base_weight, **params)
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            if yte.nunique() < 2:
                continue
            aucs.append(roc_auc_score(yte, proba))
        if aucs:
            mean_auc = float(np.mean(aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params
    return best_params, best_auc


grid = {
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "learning_rate": [0.03, 0.05],
}

best_params, best_auc = tune_xgb_hyperparams(X, y, pos_weight, grid, n_splits=3)
if best_params:
    print("Beste XGB-Parameter (Grid, AUC=%.4f):" % best_auc, best_params)

xgb = make_xgb_classifier(scale_pos_weight=pos_weight, **best_params)


def evaluate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba_all = np.full(len(y), np.nan)
    rows = []
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        mdl = clone(model)
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        proba_all[te] = p
        yhat = (p >= 0.5).astype(int)
        auc = float(roc_auc_score(yte, p)) if yte.nunique() > 1 else np.nan
        rows.append({"acc": float(accuracy_score(yte, yhat)), "auc": auc})
    return pd.DataFrame(rows), pd.Series(proba_all, index=y.index)


def evaluate_walkforward(model, X, y, min_train=WALK_MIN_TRAIN, test_window=WALK_TEST_WINDOW):
    proba_all = np.full(len(y), np.nan)
    rows = []
    start = min_train
    idx = np.arange(len(y))
    while start + test_window <= len(y):
        tr_idx = idx[:start]
        te_idx = idx[start:start + test_window]
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        mdl = clone(model)
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        proba_all[te_idx] = p
        yhat = (p >= 0.5).astype(int)
        auc = float(roc_auc_score(yte, p)) if yte.nunique() > 1 else np.nan
        rows.append({
            "start": dates[tr_idx[0]].date(),
            "end": dates[te_idx[-1]].date(),
            "acc": float(accuracy_score(yte, yhat)),
            "auc": auc,
        })
        start += test_window
    return pd.DataFrame(rows), pd.Series(proba_all, index=y.index)


models = {
    "Logit": logit,
    "RandomForest": rf,
    "XGBoost": xgb,
}

cv_results = {}
walk_results = {}
walk_probas = {}
for name, model in models.items():
    cv_df, cv_proba = evaluate_model(model, X, y, n_splits=N_SPLITS)
    cv_results[name] = cv_df
    print(f"\nCV-{name}:", cv_df.mean().round(4).to_dict())
    wf_df, wf_proba = evaluate_walkforward(model, X, y)
    walk_results[name] = wf_df
    walk_probas[name] = wf_proba
    print(f"Walk-Forward-{name}:", wf_df.mean(numeric_only=True).round(4).to_dict())

# ------------------------------------------------
# Backtest (Pair-Trade: long Asset, short Beta*Index -> excess_ret)
# Korrigierte Fees (einseitig pro Wechsel) + Mindesthaltefrist
# ------------------------------------------------
def backtest_with_min_hold(
    proba, dates, ret_series,
    thresh=0.6, fee_bps=2, slippage_bps=0,
    min_hold_days=20
):
    s = pd.Series(proba, index=dates).ffill()
    raw_sig = (s >= thresh).astype(int)

    # State machine: Entscheidung T, Ausführung T+1
    pos = []
    current = 0
    hold = 0
    for sig in raw_sig:
        if hold > 0:
            pos.append(current); hold -= 1
        else:
            if sig != current:
                current = sig
                hold = max(min_hold_days - 1, 0)
            pos.append(current)

    pos = pd.Series(pos, index=raw_sig.index).shift(1).fillna(0).astype(int)

    trades = pos.diff().fillna(pos.iloc[0]).abs()
    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    daily_costs = trades * (fee + slip)

    strat_ret = pos * ret_series - daily_costs
    equity = (1 + strat_ret).cumprod()
    bench  = (1 + ret_series).cumprod()

    n_trades = int(trades.sum())
    hold_lengths = pos.groupby((pos != pos.shift()).cumsum()).transform('size')[pos == 1]
    avg_hold = float(hold_lengths.mean()) if not hold_lengths.empty else 0.0
    turnover = float(trades.sum() / max((pos.abs().sum()), 1))
    extras = {"Trades": n_trades, "AvgHoldDays": avg_hold, "Turnover": turnover}
    return equity, bench, strat_ret, pos, extras

def metrics(equity, ret_series, rf=0.0):
    ann = 252
    cagr = equity.iloc[-1]**(ann/len(ret_series)) - 1
    vol = ret_series.std() * np.sqrt(ann)
    sharpe = ((ret_series.mean() - rf/ann) * ann) / (vol + 1e-12)
    dd_curve = equity / equity.cummax() - 1
    maxdd = dd_curve.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    downside = ret_series[ret_series < 0]
    dvol = downside.std() * np.sqrt(ann) if len(downside) else np.nan
    dsharpe = ((ret_series.mean())*ann) / (dvol + 1e-12) if pd.notna(dvol) else np.nan
    hit = (ret_series > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "DownsideSharpe": dsharpe,
            "MaxDD": maxdd, "Calmar": calmar, "Hit-Rate": hit}

proba_series = walk_probas["XGBoost"].reindex(dates).ffill()

eq, bench_ex, strat, pos, extras = backtest_with_min_hold(
    proba_series, dates, df_feat["excess_ret"].loc[dates],
    thresh=PRED_THRESH, fee_bps=FEE_BPS, slippage_bps=0, min_hold_days=MIN_HOLD_DAYS
)

m_strat = {k: round(v,4) for k,v in metrics(eq, strat).items()}
m_bench = {k: round(v,4) for k,v in metrics(bench_ex, df_feat["excess_ret"].loc[dates]).items()}
print("\nExtras:", {k: (round(v,4) if isinstance(v, float) else v) for k,v in extras.items()})
print("Strategy (Pair-Trade, excess_ret):", m_strat)
print("Benchmark (passiv excess_ret):   ", m_bench)

# Plot Equity
plt.figure(figsize=(10,6))
eq.plot(label="Strategy (Pair-Trade)")
bench_ex.plot(label="Benchmark (excess_ret Buy&Hold)")
plt.legend()
plt.title(f"Equity Curve – {TICKER} vs. SMI (Outperformance-Strategie)")
plt.tight_layout()
plt.show()

# Feature Importances (XGB)
xgb.fit(X, y)
fi = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
print("\nTop-Features:\n", fi.round(4))
