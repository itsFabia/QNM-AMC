# scripts/bias_check.py
# ------------------------------------------------------------
# QNM-AMC | Bias-Review: Ticker- & Index-Verteilung
# ------------------------------------------------------------
# Zweck:
# Prüft, ob bestimmte Ticker oder Indizes überrepräsentiert sind
# und ob alle Ticker über ausreichend lange Zeiträume abgedeckt sind.
# Erstellt einen Markdown-Report unter reports/bias_log.md.
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
from datetime import datetime

# ------------------------------
# Projektverzeichnisse
# ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = DATA_DIR / "AMC_model_input_reduced.csv"
REPORT_FILE = REPORTS_DIR / "bias_log.md"


# Hilfsfunktion zum sicheren Formatieren von Datumswerten
def fmt_date(x):
    """Gibt Datum als ISO-String zurück oder '-' bei NaT."""
    if pd.isna(x):
        return "-"
    return x.strftime("%Y-%m-%d")


def main():
    # ------------------------------
    # 1. Input prüfen und laden
    # ------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input-Datei fehlt: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Pflichtspalten prüfen – ohne Date oder Ticker keine Analyse
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("Spalten 'Date' und/oder 'Ticker' fehlen im Datensatz.")

    # Datumsformat sicher parsen; ungültige Werte zu NaT konvertieren
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Zeilen ohne Datum entfernen, sonst Fehler bei min/max
    df = df.dropna(subset=["Date"]).copy()

    # ------------------------------
    # 2. Optionale Index-/Marktspalte erkennen
    # ------------------------------
    # Manche Dateien enthalten zusätzlich Spalten wie "Index" oder "Market"
    # → wir suchen automatisch die erste passende.
    index_col = next((c for c in ["Index", "Market", "Benchmark"] if c in df.columns), None)

    # ------------------------------
    # 3. Ticker-Verteilung
    # ------------------------------
    # Wie stark ist jeder Ticker im Datensatz vertreten?
    ticker_counts = df["Ticker"].value_counts()
    ticker_ratio = (ticker_counts / ticker_counts.sum()).round(4)

    # ------------------------------
    # 4. Index-Verteilung (optional)
    # ------------------------------
    index_ratio = None
    if index_col:
        index_counts = df[index_col].value_counts(dropna=True)
        index_ratio = (index_counts / index_counts.sum()).round(4)

    # ------------------------------
    # 5. Zeitliche Abdeckung pro Ticker – nur Tage mit echtem Preis
    # ------------------------------
    # Preis-Spaltenname robust ermitteln
    price_col = None
    for c in ["Price", "Last Price", "Close", "Adj Close", "Last Price "]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("Keine Preis-Spalte gefunden (erwartet z.B. 'Price' oder 'Last Price').")

    # Preis als numerisch parsen (fehlerhafte Werte -> NaN)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Nur valide Preis-Zeilen für die Abdeckungsanalyse
    valid = df[df[price_col].notna() & (df[price_col] > 0)].copy()

    # Falls es Ticker ohne einen einzigen validen Preis gibt, wollen wir sie sehen:
    tickers_all = df["Ticker"].unique()
    tickers_valid = valid["Ticker"].unique()
    tickers_empty = sorted(set(tickers_all) - set(tickers_valid))

    # Aggregation: min/max nur über valide Preis-Tage
    span = (
        valid.groupby("Ticker", as_index=True)["Date"]
        .agg(first_with_price="min", last_with_price="max", obs_days="nunique")
    )

    # Kalenderspanne zwischen erstem und letztem Preis-Tag
    span["calendar_days"] = (span["last_with_price"] - span["first_with_price"]).dt.days + 1
    # Dauer in Jahren (auf Basis der Kalenderspanne)
    span["duration_years"] = span["calendar_days"] / 365
    # Dichte der Beobachtungen (wie voll sind die Tage zwischen min/max)
    span["density"] = (span["obs_days"] / span["calendar_days"]).round(3)

    # Sortierung: längste echte Abdeckung zuerst
    span = span.sort_values(["duration_years", "obs_days"], ascending=[False, False])

    # ------------------------------
    # 6. Markdown-Report schreiben (angepasste Tabelle)
    # ------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Bias Review – {ts}",
        f"**Input:** `{INPUT_FILE.name}`",
        "",
        "## 1. Ticker-Verteilung",
        "| Ticker | Anteil (%) |",
        "|---|---|",
    ]
    for t, r in ticker_ratio.items():  # kein Limit mehr – SMI hat nur 20
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

    # Neue Abdeckungstabelle: nur Tage mit Preis
    lines += [
        "",
        "## 3. Zeitliche Abdeckung pro Ticker (nur Tage mit gültigem Preis)",
        "| Ticker | Von | Bis | Dauer (Jahre) | Beobachtungen | Dichte |",
        "|---|---|---|---|---|---|",
    ]
    for t, row in span.iterrows():
        von = row["first_with_price"].strftime("%Y-%m-%d")
        bis = row["last_with_price"].strftime("%Y-%m-%d")
        lines.append(
            f"| {t} | {von} | {bis} | {row['duration_years']:.1f} | {int(row['obs_days'])} | {row['density']:.3f} |"
        )

    # Ticker ohne einen einzigen gültigen Preis prominent ausweisen (z. B. Pre-IPO im Beobachtungsfenster)
    if tickers_empty:
        lines += [
            "",
            "### Ticker ohne valide Preis-Historie im betrachteten Fenster",
            "",
            ", ".join(tickers_empty),
        ]

    lines += [
        "",
        "## 4. Hinweise",
        "- Abdeckung basiert nur auf Handelstagen mit gültigem Preis (> 0).",
        "- `Dichte` zeigt, wie vollständig die Tage zwischen erstem und letztem Preis abgedeckt sind.",
        "- IPOs erscheinen mit späterem Startdatum (z. B. **ALC**, **AMRZ**).",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Bias-Report gespeichert: {REPORT_FILE}")


if __name__ == "__main__":
    main()
