# ---------------------------------------------
# QNM AMC – Feature & Sanity Check
# ---------------------------------------------
# Ziel: sicherstellen, dass die Datenbasis für das ML-Modell sauber ist
# (keine Duplikate, keine NaNs, keine Zukunftsinfos, sinnvolle Korrelationen)
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. Daten einlesen ===
# Prüft automatisch auf Semikolon oder Komma als Trenner
try:
    df = pd.read_csv("../AMC_model_input.csv", sep=";", encoding="utf-8-sig")
    if df.shape[1] == 1:
        df = pd.read_csv("../AMC_model_input.csv")
except Exception as e:
    raise ValueError(f"Fehler beim Einlesen: {e}")

print(f"Anzahl Zeilen: {len(df):,}")
print(f"Spalten: {list(df.columns)[:10]} ...\n")

# === 2. Grundstruktur prüfen ===
# Datum als datetime, Duplikate erkennen
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
duplicates = df.duplicated(subset=['Date', 'Ticker']).sum()
print(f"Duplikate (Date+Ticker): {duplicates}")

# === 3. Fehlende Werte ===
# Anteil NaN je Spalte → zeigt, ob Datenquellen Lücken haben
nan_ratio = df.isna().mean().sort_values(ascending=False)
print("\nAnteil fehlender Werte (Top 10):")
print(nan_ratio.head(10))

# === 4. Outlier-Check ===
# Wir prüfen starke Ausreisser in Returns oder Volatilität
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
outlier_ratio = (np.abs(z_scores) > 5).mean().sort_values(ascending=False)
print("\nAnteil starker Ausreisser (>5 SD) (Top 10):")
print(outlier_ratio.head(10))

# === 5. Leak-Check ===
# Prüft, ob Feature-Spalten fälschlich zukünftige Infos enthalten
# Annahme: alle Feature-Namen mit '_lag' oder '.shift' sind korrekt,
# alle anderen (z. B. ohne lag) dürfen nicht nach Date+Ticker future-info zeigen
# -> Wir testen, ob Target-Werte zeitlich nach Features liegen
if 'Excess_5d_fwd' in df.columns:
    future_corr = {}
    for col in numeric_cols:
        if col != 'Excess_5d_fwd':
            corr = df[col].corr(df['Excess_5d_fwd'])
            future_corr[col] = corr
    corr_series = pd.Series(future_corr).sort_values(key=lambda x: abs(x), ascending=False)
    print("\nKorrelationen mit Target (Top 10, evtl. Hinweis auf Lookahead):")
    print(corr_series.head(10))
else:
    print("\nKein Target 'Excess_5d_fwd' gefunden – Leak-Check übersprungen.")

# === 6. Korrelationen visualisieren ===
# Schnelle Heatmap, um Cluster und doppelte Infos zu sehen
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', center=0)
plt.title("Korrelationsmatrix – numerische Features")
plt.show()

# === 7. Zusammenfassung ===
# Kurzer Überblick über potentielle Datenprobleme
print("\n--- Quick Summary ---")
print(f"- {duplicates} Duplikate gefunden.")
print(f"- {nan_ratio[nan_ratio > 0].count()} Spalten enthalten NaNs.")
print(f"- {outlier_ratio[outlier_ratio > 0].count()} Spalten mit Ausreissern.")
print("→ Wenn Korrelationen extrem hoch (>0.95), ggf. Features entfernen.")
