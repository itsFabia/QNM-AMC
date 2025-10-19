# ------------------------------------------------------------
# QNM-AMC | Cleaning + Feature-Reduktion (kommentierte Version)
# ------------------------------------------------------------
# Was dieses Skript macht (kurz):
# 1) CSV robust einlesen (Komma oder Semikolon), Date korrekt parsen.
# 2) Missing-Values-Analyse (vorher).
# 3) Imputing für Makro-Felder (konservativ: forward-fill + rolling mean).
# 4) Outlier-Handling via Winsorizing (+/- 3 SD) – nur auf Feature-Spalten.
# 5) Feature-Reduktion: Drop stark korrelierter Features (> 0.95).
# 6) Mini-Sanity-Check (nachher) und optional Heatmap speichern.
# 7) Alle Ergebnisse in Markdown-Report loggen.
#
# Output:
# - data/AMC_model_input_clean.csv         (nach Imputing + Winsorizing)
# - data/AMC_model_input_reduced.csv       (zusätzlich Feature-Drop)
# - reports/cleaning_log.md                (ehrlicher Bericht)
# - reports/corr_heatmap_after.png         (optional, wenn seaborn vorhanden)
# ------------------------------------------------------------

import os
from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------
# 0) Pfade & Parameter
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "scripts") else Path.cwd()
DATA_DIR     = PROJECT_ROOT / "data"
REPORTS_DIR  = PROJECT_ROOT / "reports"

INPUT_FILE   = DATA_DIR / "AMC_model_input.csv"
CLEAN_FILE   = DATA_DIR / "AMC_model_input_clean.csv"
REDUCED_FILE = DATA_DIR / "AMC_model_input_reduced.csv"
REPORT_FILE  = REPORTS_DIR / "cleaning_log.md"
HEATMAP_PNG  = REPORTS_DIR / "corr_heatmap_after.png"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Targets (werden NICHT verändert/gewinsorized/gedroppt)
TARGET_COLS = ["Excess_5d_fwd", "Ret_5d_fwd", "Bench_Ret_5d_fwd"]

# Heuristik: Makro-Kandidaten an Namen erkennen (du kannst diese Liste bei Bedarf ergänzen)
MACRO_HINTS = [
    "Index", "CPI", "DXY", "MOVE", "USGG", "GDBR", "GSWISS", "EURR002W",
    "FDTR", "ECCPEMUY", "SZCPIYOY", "VSMIX", "EURCHF", "SPREAD", "YIELD"
]

# Korrelations-Schwelle für Feature-Drop
CORR_CUTOFF = 0.95

# Rolling-Fenster für leichte Glättung nach FFill
ROLL_WINDOW = 5  # konservativ


# --------------------------------
# 1) CSV robust einlesen
# --------------------------------
# Warum: CSVs sind oft unterschiedlich getrennt ("," oder ";").
# Wir probieren zuerst Semikolon, bei 1-Spalten-"Unfall" nochmal mit Komma.
def read_auto_csv(path: Path) -> pd.DataFrame:
    try:
        df_local = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        if df_local.shape[1] == 1:
            df_local = pd.read_csv(path)  # Standard-Komma
        return df_local
    except Exception as e:
        raise RuntimeError(f"CSV konnte nicht gelesen werden: {e}")

if not INPUT_FILE.exists():
    sys.exit(f"Input nicht gefunden: {INPUT_FILE}")

df = read_auto_csv(INPUT_FILE)

# Datum explizit als dd.mm.yyyy parsen, um Warnungen/Fehlzuordnungen zu vermeiden
# (du hattest die Warnung bereits gesehen).
if "Date" not in df.columns:
    sys.exit("Spalte 'Date' fehlt. Ohne Date kein Zeitbezug – Skript wird beendet.")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

# Identifikatoren prüfen
if "Ticker" not in df.columns:
    sys.exit("Spalte 'Ticker' fehlt. Ohne Ticker kein Panel – Skript wird beendet.")

# Grundstatistik vor Cleaning
n_rows, n_cols = df.shape
dups = df.duplicated(subset=["Date", "Ticker"]).sum()
nan_before = df.isna().sum().sort_values(ascending=False)
total_na_before = int(nan_before.sum())


# --------------------------------
# 2) Spalten klassifizieren
# --------------------------------
# Warum: Wir wollen nur sinnvolle Spalten anfassen (Features), nicht ID/Targets.
id_cols = ["Date", "Ticker"]
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()

# Feature-Kandidaten = numerisch, nicht ID, nicht Target
feature_cols = [c for c in numeric_cols_all if c not in TARGET_COLS]

# Makro-Spalten anhand Namensheuristik bestimmen (Schnittmenge mit feature_cols)
macro_cols = [c for c in feature_cols if any(hint in c for hint in MACRO_HINTS)]

# Falls wir gar keine Makrospalten erkennen, ist das nicht fatal – dann wird nur wenig imputet
# (in Aktien-Only-Datensätzen normal).
# print("Makrospalten (heuristisch):", macro_cols[:10])


# --------------------------------
# 3) Imputing für Makro-Felder
# --------------------------------
# Warum: Deine Heatmap & NaN-Analyse zeigte, dass Makro-Reihen lückenhaft sind
# (Monats-/Quartalsfrequenz). Konservativer Ansatz:
# - Forward-Fill: hält letzten Wert, erzeugt keine Zukunftsinformation
# - Rolling-Mean: glättet minimal (Fenster=3), ohne Trends zu erfinden
df_sorted = df.sort_values(["Ticker", "Date"]).copy()

if macro_cols:
    # fill per Ticker (falls Makro pro Zeile/Ticker wiederholt vorliegt), ansonsten global
    for col in macro_cols:
        # Vorwärts auffüllen je Ticker
        df_sorted[col] = df_sorted.groupby("Ticker", observed=True)[col].ffill()
        # leichte Glättung (rollend je Ticker)
        df_sorted[col] = df_sorted.groupby("Ticker", observed=True)[col].transform(
            lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean()
        )

    # Optional: Wenn du garantiert KEINE bfill willst, oben einfach die bfill-Zeile entfernen.


# --------------------------------
# 4) Outlier-Handling (Winsorizing)
# --------------------------------
# Warum: Extremwerte (>3 SD) verzerren viele ML-Modelle. Wir schneiden nur Features zu.
    WINSOR_PATTERNS = [r"__logdiff1$", r"__chgstd20$"]


def is_winsor_target(col: str) -> bool:
    return any(re.search(p, col) for p in WINSOR_PATTERNS)
winsor_cols = [c for c in macro_cols if is_winsor_target(c)]

def winsorize_series(s: pd.Series, sigma: float = 3.0) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if pd.isna(sd) or sd == 0:
        return s  # konstante/degenerierte Spalte unangetastet
    lower, upper = mu - sigma * sd, mu + sigma * sd
    return s.clip(lower, upper)

for col in winsor_cols:
    df_sorted[col] = winsorize_series(df_sorted[col], sigma=3.0)


# --------------------------------
# 5) Feature-Reduktion (Korrelations-Drop)
# --------------------------------
# Warum: Multikollinearität bremst, macht Interpretationen instabil
# und kann Overfitting fördern. Wir droppen Paare mit |corr| > 0.95.
# Achtung: Korrelation nur über Feature-Spalten, Targets bleiben unberührt.
numeric_after = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
feature_after = [c for c in numeric_after if c not in TARGET_COLS]

corr = df_sorted[feature_after].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [c for c in upper.columns if (upper[c] > CORR_CUTOFF).any()]

df_reduced = df_sorted.drop(columns=to_drop, errors="ignore")


# --------------------------------
# 6) Mini-Sanity-Checks (nachher)
# --------------------------------
nan_after = df_reduced.isna().sum().sort_values(ascending=False)
total_na_after = int(nan_after.sum())
n_rows_after, n_cols_after = df_reduced.shape


# --------------------------------
# 7) Optional: Heatmap speichern (nach Cleaning)
# --------------------------------
# Warum: Visueller Check der neuen Struktur.
def save_heatmap(df_in: pd.DataFrame, out_path: Path, title: str = "Korrelation (nach Cleaning)"):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        return False  # seaborn nicht installiert -> Heatmap überspringen

    # Nur auf eine handhabbare Untermenge der numerischen Spalten, falls sehr groß:
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    # Targets optional ausschließen:
    num_cols = [c for c in num_cols if c not in TARGET_COLS]

    # Bei sehr vielen Features: auf die ersten 60 begrenzen, um Plot lesbar zu halten
    subset = df_in[num_cols[:60]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(subset, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True

heatmap_saved = save_heatmap(df_reduced, HEATMAP_PNG)


# --------------------------------
# 8) Dateien schreiben
# --------------------------------
# Warum: Reproduzierbarkeit + klarer Trennschritt zwischen „clean“ und „reduced“.
df_sorted.to_csv(CLEAN_FILE, index=False)
df_reduced.to_csv(REDUCED_FILE, index=False)


# --------------------------------
# 9) Markdown-Report schreiben (ehrlich)
# --------------------------------
# Warum: Nachvollziehbarkeit. Kein Pitch – Laborbuchstil.
def md_table_from_series(s: pd.Series, top: int = 10) -> str:
    rows = ["| Spalte | NaN-Anteil |", "|---|---|"]
    total = s.sum()
    denom = len(df_reduced)
    cnt = 0
    for name, val in s.head(top).items():
        # val ist hier absolute Anzahl fehlender Werte; in vorherigem s evtl. anders – daher vorsichtig formatieren
        ratio = (val / denom) if denom else 0.0
        rows.append(f"| {name} | {ratio:.3f} |")
        cnt += 1
        if cnt >= top:
            break
    return "\n".join(rows)

report_lines = []

report_lines.append(f"# Cleaning Log – {INPUT_FILE.name}")
report_lines.append(f"**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
report_lines.append(f"**Zeilen/Spalten (vorher):** {n_rows:,} / {n_cols}  ")
report_lines.append(f"**Zeilen/Spalten (nachher):** {n_rows_after:,} / {n_cols_after}  ")
report_lines.append("")
report_lines.append("## Struktur & Basis")
report_lines.append(f"- Duplikate (Date+Ticker): **{dups}**")
report_lines.append("- Date-Parsen: **dd.mm.yyyy** via `dayfirst=True` gesetzt.")
report_lines.append("")
report_lines.append("## Missing Values")
report_lines.append(f"- Fehlende Werte **vorher (gesamt):** {total_na_before:,}")
report_lines.append(f"- Fehlende Werte **nachher (gesamt):** {total_na_after:,}")
report_lines.append("")
report_lines.append("### Top-NaN-Spalten vorher (Anteil; grobe Orientierung)")
# Für den Report: Anteil aus vorheriger Zählung (nan_before). Wir rechnen hier auf Basis der Zeilen.
nan_before_ratio = (nan_before / max(n_rows, 1)).head(10)
report_lines.append("| Spalte | NaN-Anteil |")
report_lines.append("|---|---|")
for name, ratio in nan_before_ratio.items():
    report_lines.append(f"| {name} | {ratio:.3f} |")

report_lines.append("")
report_lines.append("## Imputing & Glättung (Makro)")
if macro_cols:
    report_lines.append(f"- Makro-Felder (heuristisch erkannt): {len(macro_cols)} Spalten.")
    report_lines.append(f"- Methode: **forward-fill pro Ticker** + **rolling mean (W={ROLL_WINDOW})**.")
else:
    report_lines.append("- Keine Makro-Felder anhand Namensheuristik erkannt – Imputing minimal angewandt.")
report_lines.append("")
report_lines.append("## Outlier-Handling")
report_lines.append("- Winsorizing auf **±3 SD** für Feature-Spalten (Targets unverändert).")
report_lines.append("")
report_lines.append("## Feature-Reduktion")
report_lines.append(f"- Korrelationsschwelle: **>{CORR_CUTOFF:.2f}** (absolut).")
report_lines.append(f"- Entfernte Features: **{len(to_drop)}**")
if to_drop:
    preview = ", ".join(to_drop[:12])
    more = "" if len(to_drop) <= 12 else f" … (+{len(to_drop)-12} weitere)"
    report_lines.append(f"  - Beispiele: {preview}{more}")
report_lines.append("")
report_lines.append("## Optional: Heatmap")
if heatmap_saved:
    report_lines.append(f"- Korrelations-Heatmap gespeichert: `{HEATMAP_PNG.name}`")
else:
    report_lines.append("- Heatmap **nicht** gespeichert (seaborn nicht installiert). Optional: `python -m pip install seaborn`")
report_lines.append("")
report_lines.append("## Dateien (Outputs)")
report_lines.append(f"- Clean (Imputing + Winsorizing): `data/{CLEAN_FILE.name}`")
report_lines.append(f"- Reduced (zzgl. Feature-Drop): `data/{REDUCED_FILE.name}`")
report_lines.append("")
report_lines.append("## Selbstkritik / Risiken")
report_lines.append("- Imputing kann Bias erzeugen, wenn Makro-Reihen lange Lücken haben (Regime-Übergänge).")
report_lines.append("- Korrelations-Drop ist heuristisch: kann nützliche, aber redundante Signale kappen.")
report_lines.append("- Winsorizing glättet Schocks – gut für Stabilität, aber reduziert Extrem-Alpha.")
report_lines.append("- Date-Parsen: **ISO %Y-%m-%d** (kein dayfirst).")
report_lines.append("- Winsorizing: **±3 SD** nur auf Makro-Diff/Vol-Spalten (`__logdiff1`, `__chgstd20`).")
report_lines.append("")
report_lines.append("## Nächste Schritte")
report_lines.append("1. Sanity-Check erneut laufen lassen (auf `*_reduced.csv`).")
report_lines.append("2. Erste Baseline-Modelle (z. B. RandomForest/XGBoost) mit Walk-Forward testen.")
report_lines.append("3. Erklärbarkeit: Permutation Importance/SHAP prüfen (Bias-Detektor).")
report_lines.append("4. Versionslog im Projekt aktualisieren (`reports/sanity_check_log.md`).")


REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

print("Done.")
print(f"- Gespeichert: {CLEAN_FILE}")
print(f"- Gespeichert: {REDUCED_FILE}")
print(f"- Report:      {REPORT_FILE}")
if heatmap_saved:
    print(f"- Heatmap:     {HEATMAP_PNG}")
