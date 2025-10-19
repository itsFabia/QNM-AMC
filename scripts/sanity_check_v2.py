# ------------------------------------------------------------
# QNM-AMC | Sanity Check V2 – nach Cleaning & Feature-Reduktion
# ------------------------------------------------------------
# Ziel: prüfen, ob das Cleaning (Imputing, Winsorizing, Feature-Drop)
#       die Daten konsistent und modellfähig hinterlassen hat.
# Verglichen mit V1: weniger Spalten, weniger NaNs, stabile Korrelation.
# ------------------------------------------------------------
# QNM-AMC | Sanity Check (robust & knapp)
# Läuft stabil, egal von wo gestartet (setzt ROOT aus __file__)

from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Pfade robust setzen ----------
ROOT     = Path(__file__).resolve().parents[1]         # .../QNM-AMC
DATA     = ROOT / "data"
REPORTS  = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)

INPUT    = DATA / "AMC_model_input_reduced.csv"        # <- bei Bedarf ändern
HEATMAP  = REPORTS / "corr_heatmap_v2.png"
REPORT   = REPORTS / "sanity_check_v2_log.md"

if not INPUT.exists():
    raise FileNotFoundError(f"Input nicht gefunden: {INPUT}")

# ---------- Daten laden ----------
df = pd.read_csv(INPUT)
print(f"Zeilen: {len(df):,}, Spalten: {len(df.columns)}")
print("Spaltenbeispiele:", list(df.columns)[:10])

# Datum sauber (ISO) parsen – Warnungen vermeiden
if "Date" not in df.columns or "Ticker" not in df.columns:
    raise ValueError("Spalten 'Date' und/oder 'Ticker' fehlen.")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

# ---------- Struktur ----------
dups = df.duplicated(subset=["Date", "Ticker"]).sum()
print("Duplikate (Date+Ticker):", dups)

# ---------- NaNs ----------
nan_ratio = df.isna().mean().sort_values(ascending=False)
print("\nTop 10 fehlende Werte (pro Spalte):\n", nan_ratio.head(10))

# ---------- Ausreisser (>5 SD) ----------
numeric_cols = df.select_dtypes(include=[np.number]).columns
z = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
outlier_ratio = (z.abs() > 5).mean().sort_values(ascending=False)
print("\nAnteil starker Ausreisser (>5 SD):\n", outlier_ratio.head(10))

# ---------- Leak-Check (Korr mit Target) ----------
target = "Excess_5d_fwd"
if target in df.columns:
    corr_series = (
        df[numeric_cols]
        .corrwith(df[target])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    print("\nTop 10 Korrelationen mit Target:\n", corr_series.head(10))
else:
    print("\nKein Target 'Excess_5d_fwd' gefunden – Leak-Check übersprungen.")

# ---------- Heatmap (optional) ----------
def save_heatmap_safe(df_in, out_path):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("seaborn nicht installiert – Heatmap übersprungen.")
        return
    num = df_in.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        print("Keine numerischen Spalten – Heatmap übersprungen.")
        return
    subset = df_in[num[:60]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(subset, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Korrelationsmatrix – numerische Features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Heatmap gespeichert: {out_path}")

save_heatmap_safe(df, HEATMAP)

# ---------- Markdown-Report (ohne tabulate) ----------
def series_to_md_table(s, n=10, col="Wert", fmt="{:.6f}"):
    lines = ["| Spalte | " + col + " |", "|---|---|"]
    for k, v in s.head(n).items():
        try: vv = fmt.format(float(v))
        except Exception: vv = str(v)
        lines.append(f"| {k} | {vv} |")
    return "\n".join(lines)

report = []
report += [f"# Sanity Check – {INPUT.name}", ""]
report += ["## 1. Struktur",
           f"- Zeilen: {len(df):,}",
           f"- Spalten: {len(df.columns)}",
           f"- Duplikate (Date+Ticker): {dups}",
           "- Date-Parsing: ISO (%Y-%m-%d)"]
report += ["", "## 2. Fehlende Werte",
           f"- Spalten mit NaNs: {int((nan_ratio>0).sum())}",
           f"- Durchschnittlicher NaN-Anteil: {nan_ratio.mean():.3f}",
           "", "### Top 10 NaN-Spalten",
           series_to_md_table(nan_ratio, 10, col="NaN-Anteil")]
report += ["", "## 3. Ausreisser (>5σ)",
           f"- Spalten mit Ausreissern: {int((outlier_ratio>0).sum())}",
           f"- Ø Ausreisser-Anteil: {outlier_ratio.mean():.6f}",
           "", "### Top 10 Ausreisser-Spalten",
           series_to_md_table(outlier_ratio, 10, col="Anteil")]
report += ["", "## 4. Leak-Check (Korrelation mit Target)"]
if target in df.columns:
    c = (
        df[numeric_cols]
        .corrwith(df[target])
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    report += [series_to_md_table(c, 10, col="Korrelation")]
else:
    report += ["_Target 'Excess_5d_fwd' nicht gefunden – übersprungen._"]
report += ["", "## 5. Heatmap",
           f"- Datei: `reports/{HEATMAP.name}`" if HEATMAP.exists() else "- (nicht erzeugt)"]

REPORT.write_text("\n".join(report), encoding="utf-8")
print(f"\nReport gespeichert: {REPORT}")
