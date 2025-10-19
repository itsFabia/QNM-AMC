# scripts/bias_check.py
# ------------------------------------------------------------
# QNM-AMC | Bias-Review
# ------------------------------------------------------------
# Zweck:
# - Prüft strukturellen Bias (Ticker/Index-Verteilung, Preisabdeckung)
# - Misst Feature-Level-Bias über Ticker hinweg
# - Erkennt späte IPOs & kurze Historien
# - Bestimmt empfohlenen Trainingsstart (≥90% Ticker aktiv)
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = DATA_DIR / "AMC_model_input_reduced.csv"
REPORT_FILE = REPORTS_DIR / "bias_log.md"

PRICE_CANDIDATES = ["Price", "Last Price", "PX_LAST", "Close"]
FEATURE_CANDIDATES = [
    "MktCap", "DivYld12m", "Volume", "Return_MA5", "Return_STD5",
    "Return_MA10", "Return_STD10", "Return_MA20", "Return_STD20",
    "Return_raw", "Return_lag1", "Return_lag5", "Return_Z20"
]

def fmt_date(x):
    if pd.isna(x):
        return "-"
    return pd.Timestamp(x).strftime("%Y-%m-%d")

def find_price_column(df):
    for c in PRICE_CANDIDATES:
        if c in df.columns:
            return c
    return None

def main():
    # ------------------------------
    # 1. Load data
    # ------------------------------
    df = pd.read_csv(INPUT_FILE)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("Spalten 'Date' oder 'Ticker' fehlen.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    price_col = find_price_column(df)
    if not price_col:
        raise ValueError("Keine Preisspalte gefunden.")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # ------------------------------
    # 2. Verteilungen
    # ------------------------------
    ticker_counts = df["Ticker"].value_counts()
    ticker_ratio = (ticker_counts / ticker_counts.sum()).round(4)
    index_col = next((c for c in ["Index", "Market", "Benchmark"] if c in df.columns), None)
    index_ratio = None
    if index_col:
        index_counts = df[index_col].value_counts()
        index_ratio = (index_counts / index_counts.sum()).round(4)

    # ------------------------------
    # 3. Zeitliche Abdeckung
    # ------------------------------
    valid = df[df[price_col].notna() & (df[price_col] > 0)].copy()
    span = (
        valid.groupby("Ticker")["Date"]
        .agg(first_with_price="min", last_with_price="max", obs_days="nunique")
    )
    span["calendar_days"] = (span["last_with_price"] - span["first_with_price"]).dt.days + 1
    span["duration_years"] = span["calendar_days"] / 365

    global_first = valid["Date"].min()
    global_last = valid["Date"].max()
    ipo_like = span.index[span["first_with_price"] > global_first].tolist()
    too_short = span.index[span["obs_days"] < 250].tolist()  # <1 Jahr

    # 90%-Trainingsfenster
    active_by_day = valid.groupby("Date")["Ticker"].nunique().sort_index()
    threshold = int(round(0.9 * active_by_day.max()))
    train_start_reco = active_by_day.index[active_by_day >= threshold][0]

    # ------------------------------
    # 4. Feature-Level-Bias
    # ------------------------------
    feature_tables = []
    for f in FEATURE_CANDIDATES:
        if f not in df.columns:
            continue
        s = pd.to_numeric(df[f], errors="coerce")
        valid_feat = df.loc[s.notna(), ["Ticker", f]].copy()
        if valid_feat.empty:
            continue
        stats = (
            valid_feat.groupby("Ticker")[f]
            .agg(mean="mean", std="std", min="min", max="max", median="median", non_null="count")
            .round(5)
        )
        g_mean, g_std = s.mean(skipna=True), s.std(skipna=True)
        stats["mean_shift_sigma"] = ((stats["mean"] - g_mean) / g_std).round(2) if g_std else np.nan
        stats["cv"] = (stats["std"] / stats["mean"]).replace([np.inf, -np.inf], np.nan).round(3)
        feature_tables.append((f, stats, {"global_mean": g_mean, "global_std": g_std}))

    # ------------------------------
    # 5. Report
    # ------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Bias Review – {ts}",
        f"**Input:** `{INPUT_FILE.name}`",
        "",
        "## 1. Überblick",
        f"- **Von:** {fmt_date(global_first)}",
        f"- **Bis:** {fmt_date(global_last)}",
        f"- **Empfohlener Trainingsstart (≥90% aktiv):** {fmt_date(train_start_reco)}",
        f"- **Ticker gesamt:** {df['Ticker'].nunique()}",
        "",
        "## 2. Ticker-Verteilung",
        "| Ticker | Anteil (%) |",
        "|---|---|",
    ] + [f"| {t} | {r*100:.2f} |" for t, r in ticker_ratio.items()]

    if index_ratio is not None:
        lines += [
            "",
            "## 3. Index-Verteilung",
            "| Index | Anteil (%) |",
            "|---|---|",
        ] + [f"| {i} | {r*100:.2f} |" for i, r in index_ratio.items()]

    lines += [
        "",
        "## 4. Zeitliche Abdeckung (gültige Preise)",
        "| Ticker | Von | Bis | Jahre | Beobachtungen |",
        "|---|---|---|---|---|",
    ]
    for t, row in span.iterrows():
        lines.append(
            f"| {t} | {fmt_date(row['first_with_price'])} | {fmt_date(row['last_with_price'])} | "
            f"{row['duration_years']:.2f} | {int(row['obs_days'])} |"
        )

    if ipo_like:
        lines += ["", "**Späte IPOs:** " + ", ".join(ipo_like)]
    if too_short:
        lines += ["", "**Ticker mit <250 Beobachtungen:** " + ", ".join(too_short)]

    lines += [
        "",
        "## 5. Feature-Level-Bias",
        "_Hinweis: mean_shift_sigma = Abweichung des Ticker-Mittels vom globalen Mittel in σ._",
    ]
    for f, stats, meta in feature_tables:
        lines += [
            "",
            f"### Feature: `{f}` (global mean: {meta['global_mean']:.5f}, std: {meta['global_std']:.5f})",
            "| Ticker | mean | std | min | max | median | non_null | mean_shift_sigma | cv |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t, r in stats.iterrows():
            lines.append(
                f"| {t} | {r['mean']:.5f} | {r['std']:.5f} | {r['min']:.5f} | {r['max']:.5f} | "
                f"{r['median']:.5f} | {int(r['non_null'])} | "
                f"{'' if pd.isna(r['mean_shift_sigma']) else r['mean_shift_sigma']:.2f} | "
                f"{'' if pd.isna(r['cv']) else r['cv']:.3f} |"
            )

    lines += [
        "",
        "## 6. Empfehlungen",
        "- Cross-Validation: Rolling/Expanding Window Splits (keine zufälligen).",
        "- Trainingsfenster: ab empfohlenem Start oder 2021-01-01 für Vergleichbarkeit.",
        "- Bei starker mean_shift_sigma-Abweichung: Feature-Standardisierung pro Index.",
        "- Späte IPOs und kurze Historien ggf. separat modellieren.",
        "",
        f"**Preisspalte genutzt:** `{price_col}` (gültig: >0)."
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Bias-Report gespeichert: {REPORT_FILE}")

if __name__ == "__main__":
    main()
