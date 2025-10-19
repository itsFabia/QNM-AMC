# scripts/bias_check.py
# ------------------------------------------------------------
# QNM-AMC | Bias-Review: Ticker- & Index-Verteilung
# ------------------------------------------------------------
# Prüft, ob bestimmte Ticker oder Indizes überrepräsentiert sind.
# Erstellt Markdown-Report unter reports/bias_log.md
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
from datetime import datetime

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
INPUT_FILE = DATA_DIR / "AMC_model_input_reduced.csv"
REPORT_FILE = REPORTS_DIR / "bias_log.md"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input-Datei fehlt: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Prüfen, ob notwendige Spalten existieren
    required_cols = ["Ticker", "Date"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt im Datensatz.")

    # Index-Spalte suchen (optional)
    index_col = None
    for candidate in ["Index", "Market", "Benchmark"]:
        if candidate in df.columns:
            index_col = candidate
            break

    # ------------------------------
    # 1. Ticker-Verteilung
    # ------------------------------
    ticker_counts = df["Ticker"].value_counts()
    ticker_ratio = (ticker_counts / ticker_counts.sum()).round(4)

    # ------------------------------
    # 2. Index-Verteilung (falls vorhanden)
    # ------------------------------
    index_ratio = None
    if index_col:
        index_counts = df[index_col].value_counts()
        index_ratio = (index_counts / index_counts.sum()).round(4)

    # ------------------------------
    # 3. Zeitliche Abdeckung pro Ticker
    # ------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    ticker_years = (
        df.groupby("Ticker")["Date"]
        .agg(["min", "max"])
        .assign(duration_years=lambda x: (x["max"] - x["min"]).dt.days / 365)
        .sort_values("duration_years", ascending=False)
    )

    # ------------------------------
    # 4. Markdown-Report schreiben
    # ------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Bias Review – {ts}\n",
        f"**Input:** `{INPUT_FILE.name}`\n",
        "",
        "## 1. Ticker-Verteilung",
        "| Ticker | Anteil (%) |",
        "|---|---|",
    ]
    for t, r in ticker_ratio.head(15).items():
        lines.append(f"| {t} | {r * 100:.2f} |")

    if index_ratio is not None:
        lines += [
            "",
            "## 2. Index-Verteilung",
            f"**Spalte:** `{index_col}`",
            "| Index | Anteil (%) |",
            "|---|---|",
        ]
        for i, r in index_ratio.items():
            lines.append(f"| {i} | {r * 100:.2f} |")

    lines += [
        "",
        "## 3. Zeitliche Abdeckung pro Ticker",
        "| Ticker | Von | Bis | Dauer (Jahre) |",
        "|---|---|---|---|",
    ]
    for idx, row in ticker_years.head(10).iterrows():
        lines.append(
            f"| {idx} | {row['min'].date()} | {row['max'].date()} | {row['duration_years']:.1f} |"
        )

    lines += [
        "",
        "## 4. Hinweise",
        "- Unausgeglichene Verteilungen können das Modell verzerren.",
        "- Falls einzelne Märkte stark dominieren, separate Trainings-Sets oder Gewichtungen prüfen.",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Bias-Report gespeichert: {REPORT_FILE}")

if __name__ == "__main__":
    main()
