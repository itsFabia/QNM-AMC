"""Outperformance-Strategie für Aktienindizes.

Dieses Modul kombiniert drei Schritte:
1. Ranking langfristiger Outperformer innerhalb eines Index (Total-Return).
2. Maschinelles Lernen zur kurzfristigen Vorhersage der Markt-Outperformance.
3. Einen einfachen Backtest, der die Vorhersagen in eine Handelslogik überführt.

Die Funktionen sind so aufgebaut, dass sie als Werkzeug genutzt oder per
Kommandozeile ausgeführt werden können. Standardmäßig wird der Schweizer
SMI verwendet (siehe ``top_performers_SMI``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from top_performers import compute_top_outperformers
from top_performers_SMI import SMI_BENCH, SMI_TICKERS


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Datenaufbereitung
# ---------------------------------------------------------------------------

def _flatten_columns(columns: Iterable) -> List[str]:
    """Bringt MultiIndex-Spalten (z. B. von Yahoo Finance) in String-Form."""
    if isinstance(columns, pd.MultiIndex):
        return ["_".join(str(x) for x in col if x not in (None, "")) for col in columns.values]
    return [str(c) for c in columns]


def _pick(colnames: Sequence[str], candidates: Sequence[str | None]) -> str | None:
    """Wählt den ersten passenden Spaltennamen – robust gegen Suffixe."""
    cols = [str(c).lower().replace(" ", "_") for c in colnames]
    for cand in candidates:
        if not cand:
            continue
        cand_l = cand.lower()
        if cand_l in cols:
            return cols[cols.index(cand_l)]
    for cand in candidates:
        if not cand:
            continue
        cand_l = cand.lower()
        for col in cols:
            if col.endswith(cand_l):
                return col
    return None


def _normalize_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardisiert Yahoo-Download auf die üblichen OHLC-Spalten."""
    df = raw.copy()
    df.columns = _flatten_columns(df.columns)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    t = ticker.lower()
    out: dict[str, pd.Series] = {}

    adj_col = _pick(df.columns, ["adj_close", "adjclose", f"adj_close_{t}", f"adjclose_{t}"])
    close_col = _pick(df.columns, ["close", f"close_{t}"])
    if adj_col is not None:
        out["adj_close"] = df[adj_col]
    if close_col is not None:
        out["close"] = df[close_col]
    for key in ("open", "high", "low", "volume"):
        col = _pick(df.columns, [key, f"{key}_{t}"])
        if col is not None:
            out[key] = df[col]

    if "adj_close" not in out and "close" not in out:
        raise KeyError(f"{ticker}: keine Adj/Close-Spalte gefunden (Spalten: {list(df.columns)})")

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Datenklassen für strukturierte Rückgaben
# ---------------------------------------------------------------------------

@dataclass
class LongTermSelection:
    ranking: pd.DataFrame
    benchmark_total_return: pd.Series


@dataclass
class MLData:
    features: pd.DataFrame
    target: pd.Series
    dates: pd.DatetimeIndex
    excess_returns: pd.Series


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    strategy_returns: pd.Series
    positions: pd.Series
    extras: dict


@dataclass
class EvaluationSummary:
    cv_metrics: pd.DataFrame
    backtest: BacktestResult
    strategy_metrics: dict
    benchmark_metrics: dict


# ---------------------------------------------------------------------------
# Kernfunktionen des Werkzeugs
# ---------------------------------------------------------------------------

def rank_long_term_outperformers(
    index_members: Sequence[str],
    benchmark_ticker: str,
    start: str = "2010-01-01",
    lookback_years: int = 10,
    top_n: int = 5,
) -> LongTermSelection:
    """Bestimmt die langfristigen Outperformer eines Index gegenüber der Benchmark."""
    ranking, bench_tr = compute_top_outperformers(
        index_members=list(index_members),
        benchmark_ticker=benchmark_ticker,
        start=start,
        lookback_years=lookback_years,
        top_n=top_n,
    )
    return LongTermSelection(ranking=ranking, benchmark_total_return=bench_tr)


def prepare_ml_dataset(
    asset_ticker: str,
    benchmark_ticker: str,
    start: str,
    roll_beta: int = 60,
    feature_windows: Sequence[int] = (3, 5, 10, 20),
) -> MLData:
    """Erzeugt ein Feature-Set zur Vorhersage von kurzfristiger Outperformance."""
    asset_raw = yf.download(asset_ticker, start=start, auto_adjust=True, progress=False)
    bench_raw = yf.download(benchmark_ticker, start=start, auto_adjust=True, progress=False)
    if asset_raw.empty:
        raise ValueError(f"Keine Kursdaten für {asset_ticker}")
    if bench_raw.empty:
        raise ValueError(f"Keine Kursdaten für {benchmark_ticker}")

    asset = _normalize_ohlc(asset_raw, asset_ticker)
    bench = _normalize_ohlc(bench_raw, benchmark_ticker)

    asset_close = (asset["adj_close"] if "adj_close" in asset.columns else asset["close"]).rename("asset")
    bench_close = (bench["adj_close"] if "adj_close" in bench.columns else bench["close"]).rename("bench")

    data = pd.concat([asset_close, bench_close], axis=1).dropna()
    data["ret_asset"] = data["asset"].pct_change()
    data["ret_bench"] = data["bench"].pct_change()

    cov = data["ret_asset"].rolling(roll_beta).cov(data["ret_bench"])
    var = data["ret_bench"].rolling(roll_beta).var() + 1e-12
    data["beta"] = cov / var
    data["excess_ret"] = data["ret_asset"] - data["beta"] * data["ret_bench"]
    data["target"] = (data["excess_ret"].shift(-1) > 0).astype(int)

    features = data.copy()
    for window in feature_windows:
        features[f"ret_asset_{window}d"] = data["asset"].pct_change(window)
        features[f"ret_bench_{window}d"] = data["bench"].pct_change(window)
        features[f"ret_excess_{window}d"] = features[f"ret_asset_{window}d"] - features[f"ret_bench_{window}d"]
        features[f"vol_asset_{window}d"] = data["ret_asset"].rolling(window).std()
        features[f"vol_bench_{window}d"] = data["ret_bench"].rolling(window).std()
        roll_mean = data["excess_ret"].rolling(window).mean()
        roll_std = data["excess_ret"].rolling(window).std() + 1e-9
        features[f"z_excess_{window}d"] = (data["excess_ret"] - roll_mean) / roll_std
        features[f"beta_{window}d"] = data["beta"].rolling(window).mean()
    features["dow"] = features.index.dayofweek
    features = pd.get_dummies(features, columns=["dow"], drop_first=True)
    features = features.dropna()

    drop_cols = ["asset", "bench", "ret_asset", "ret_bench", "beta", "excess_ret", "target"]
    feature_cols = [c for c in features.columns if c not in drop_cols]

    X = features[feature_cols]
    y = features["target"].astype(int)
    dates = features.index

    return MLData(features=X, target=y, dates=dates, excess_returns=features["excess_ret"])  # type: ignore[index]


def time_series_cross_val(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Zeitreihen-kompatible Cross-Validation mit fortlaufenden Splits."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    proba = pd.Series(np.nan, index=y.index)
    rows = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        p = fitted.predict_proba(X_test)[:, 1]
        proba.iloc[test_idx] = p
        y_pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "auc": float(roc_auc_score(y_test, p)),
            }
        )

    return pd.DataFrame(rows), proba


def backtest_strategy(
    probabilities: pd.Series,
    dates: pd.DatetimeIndex,
    excess_returns: pd.Series,
    threshold: float = 0.52,
    fee_bps: float = 2.0,
    min_hold_days: int = 10,
) -> BacktestResult:
    """Einfache Handelslogik: long, wenn Modell hohe Outperformance-PW erwartet."""
    signal = probabilities.reindex(dates).ffill().fillna(0.0)
    raw_position = (signal >= threshold).astype(int)

    positions = []
    current = 0
    hold = 0
    for value in raw_position:
        if hold > 0:
            positions.append(current)
            hold -= 1
            continue
        if value != current:
            current = value
            hold = max(min_hold_days - 1, 0)
        positions.append(current)

    pos_series = pd.Series(positions, index=raw_position.index).shift(1).fillna(0).astype(int)
    trades = pos_series.diff().fillna(pos_series.iloc[0]).abs()
    costs = trades * (fee_bps / 10_000.0)

    strat_returns = pos_series * excess_returns.loc[dates] - costs
    equity_curve = (1 + strat_returns).cumprod()
    benchmark_curve = (1 + excess_returns.loc[dates]).cumprod()

    hold_lengths = pos_series.groupby((pos_series != pos_series.shift()).cumsum()).transform("size")[pos_series == 1]
    extras = {
        "Trades": int(trades.sum()),
        "AverageHoldDays": float(hold_lengths.mean()) if not hold_lengths.empty else 0.0,
        "Turnover": float(trades.sum() / max(pos_series.abs().sum(), 1)),
    }

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        strategy_returns=strat_returns,
        positions=pos_series,
        extras=extras,
    )


def performance_metrics(equity_curve: pd.Series, returns: pd.Series) -> dict:
    """Berechnet gängige Kennzahlen (CAGR, Volatilität, Sharpe etc.)."""
    ann = 252
    cagr = equity_curve.iloc[-1] ** (ann / len(returns)) - 1
    vol = returns.std() * np.sqrt(ann)
    sharpe = returns.mean() / (vol + 1e-12) * ann
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(ann) if len(downside) else np.nan
    downside_sharpe = (returns.mean() * ann) / (downside_vol + 1e-12) if pd.notna(downside_vol) else np.nan
    hit_rate = (returns > 0).mean()

    return {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "DownsideSharpe": float(downside_sharpe) if pd.notna(downside_sharpe) else np.nan,
        "MaxDrawdown": float(max_dd),
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "HitRate": float(hit_rate),
    }


def build_strategy(
    index_members: Sequence[str],
    benchmark_ticker: str,
    start: str = "2010-01-01",
    lookback_years: int = 10,
    top_n: int = 5,
    roll_beta: int = 60,
    n_splits: int = 5,
    threshold: float = 0.52,
    fee_bps: float = 2.0,
    min_hold_days: int = 10,
) -> tuple[LongTermSelection, EvaluationSummary]:
    """Durchläuft das komplette Tool von der Selektion bis zum Backtest."""
    selection = rank_long_term_outperformers(
        index_members=index_members,
        benchmark_ticker=benchmark_ticker,
        start=start,
        lookback_years=lookback_years,
        top_n=top_n,
    )
    top_ticker = selection.ranking.index[0]

    ml_data = prepare_ml_dataset(
        asset_ticker=top_ticker,
        benchmark_ticker=benchmark_ticker,
        start=start,
        roll_beta=roll_beta,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    cv_metrics, probabilities = time_series_cross_val(model, ml_data.features, ml_data.target, n_splits=n_splits)

    backtest = backtest_strategy(
        probabilities=probabilities,
        dates=ml_data.dates,
        excess_returns=ml_data.excess_returns,
        threshold=threshold,
        fee_bps=fee_bps,
        min_hold_days=min_hold_days,
    )

    strategy_metrics = performance_metrics(backtest.equity_curve, backtest.strategy_returns)
    benchmark_metrics = performance_metrics(backtest.benchmark_curve, ml_data.excess_returns.loc[ml_data.dates])

    summary = EvaluationSummary(
        cv_metrics=cv_metrics,
        backtest=backtest,
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
    )
    return selection, summary


# ---------------------------------------------------------------------------
# Kommandozeilenausführung
# ---------------------------------------------------------------------------

def _format_metrics(metrics: dict) -> str:
    return ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())


def main() -> None:
    selection, summary = build_strategy(
        index_members=SMI_TICKERS,
        benchmark_ticker=SMI_BENCH,
    )

    print("Top Outperformer (10 Jahre):")
    print(selection.ranking)
    top_ticker = selection.ranking.index[0]
    print(f"\nML-Analyse für kurzfristige Outperformance: {top_ticker}")
    print(summary.cv_metrics.describe().round(4))

    print("\nBacktest-Zusammenfassung:")
    print("Extras:", summary.backtest.extras)
    print("Strategie:", _format_metrics(summary.strategy_metrics))
    print("Benchmark:", _format_metrics(summary.benchmark_metrics))


if __name__ == "__main__":  # pragma: no cover - CLI Einstieg
    main()
