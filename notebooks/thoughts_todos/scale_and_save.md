# scale_and_save.py – Feature Standardisierung pro Split & Ticker  
**Datum:** 2025-10-22  
**Autorin:** Fabia Holzer  

---

## 1. Überblick

Das Skript `scale_and_save.py` führt die **Standardisierung (log1p + z-Score)** der Features im AMC-Datensatz durch – **pro Split und pro Ticker**.  
Es nutzt die Split-Indizes, die zuvor mit `train_test_split_rolling.py` erstellt wurden, und erzeugt daraus für jedes Zeitfenster ein **train/test-Paar** mit korrekt skalierten Features.  

Ziel:  
eine **leakagefreie, reproduzierbare Datenbasis** schaffen, auf der Machine-Learning-Modelle (z. B. Random Forest, LightGBM, XGBoost) fair trainiert und validiert werden können.

---

## 2. Motivation

In Finanzzeitreihen schwanken Grössenordnungen stark:  
- Aktienpreise können von 10 CHF bis 1000 CHF reichen.  
- Handelsvolumen variiert um mehrere Grössenordnungen.  
- Makro-Indikatoren (z. B. Zinsen, CPI) liegen auf völlig anderen Skalen.  

Ein Modell ohne Normierung wird dadurch instabil: grosse Werte dominieren, kleine verlieren Einfluss.  
Zudem darf bei Zeitreihen kein **Daten-Leakage** entstehen – Statistiken aus der Zukunft dürfen **niemals** im Training landen.  

Darum:  
→ Standardisierung **innerhalb jedes Splits** (Train separat, Test danach skaliert mit Train-Stats)  
→ Transformation **tickerbasiert**, da jedes Unternehmen seine eigene Verteilung hat.

---

## 3. Ablauf & Logik

### Schritt 1 – Einlesen der Daten  
- Input: `AMC_model_input_reduced.csv`  
- Entfernt Zeilen mit `Price <= 0`.  
- Wandelt Datumsspalte in `datetime` um.  
- Ermittelt automatisch alle **numerischen Feature-Spalten** (optional durch Include/Exclude steuerbar).

---

### Schritt 2 – Laden der Split-Indizes  
- Liest die `.pkl`-Dateien aus dem vorherigen Schritt:  
splits_log_idx_000_train.pkl
splits_log_idx_000_test.pkl
- - Jeder Index zeigt genau, welche Zeilen zum Train- bzw. Test-Fenster gehören.  
- Insgesamt werden 57 Splits verarbeitet (2021–2025, Rolling Window).

---

### Schritt 3 – Transformationen pro Split

#### a) Log-Transformation (`log1p`)
Zur Reduktion von Schiefe und Extremwerten:
- Automatisch für alle stark schiefen und nicht-negativen Features (`--auto-log1p`).  
- Zusätzlich manuell definierbar über `--log1p-cols`.  
Beispiel: `Volume`, `MktCap`, `Price`, `Return_STD5`, `Return_STD20`, ...

Formel:
\[
x' = \log(1 + x)
\]

---

#### b) Z-Score-Normalisierung (pro Ticker, pro Split)
Innerhalb jedes Splits und jedes Tickers:
- Berechne **Mittelwert** und **Standardabweichung** der Features im **Train-Split**.  
- Skaliere damit Train und Test:
\[
z = \frac{x - \mu_\text{Train}}{\sigma_\text{Train}}
\]

Diese Vorgehensweise garantiert:
- Kein Datenleck vom Test in den Train.  
- Faire, zeitlich korrekte Standardisierung.  
- Robustheit gegenüber firmenspezifischen Level-Unterschieden.

---

### Schritt 4 – NaN-Imputation
Falls Standardisierung zu NaNs führt (z. B. durch fehlende Daten oder `std = 0`):
- Ersetzt durch 0, falls `--impute-na 0` gesetzt wurde.  
- Neutraler Wert (keine Verzerrung der Verteilung).

---

### Schritt 5 – Speicherung pro Split
Für jeden Split werden 3 Dateien erzeugt:

| Dateityp | Inhalt | Zweck |
|-----------|---------|--------|
| `scaled_train_###.parquet` | Train-Split, vollständig skaliert | Training des Modells |
| `scaled_test_###.parquet`  | Test-Split, mit Train-Stats skaliert | Validierung |
| `scaler_stats_###.parquet` | Mittelwerte & Standardabweichungen pro Feature & Ticker | Reproduzierbarkeit |

Zusätzlich:  
`meta_###.json` → kleine Übersicht mit Split-ID, Featurezahl, Log-Transformationen etc.

---

## 4. Technische Struktur

| Komponente | Aufgabe |
|-------------|----------|
| `auto_feature_cols()` | erkennt automatisch numerische Feature-Spalten |
| `choose_log1p_cols()` | wählt Spalten für log1p-Transformation nach Schiefe |
| `compute_group_stats()` | berechnet Mittelwert & Std pro Ticker & Feature im Train |
| `zscore_by_ticker()` | führt Z-Transformation pro Ticker durch |
| `infer_idx_pattern()` | erkennt Pfad & Präfix der Split-Indizes |
| `list_split_files()` | listet vorhandene Split-IDs (000–056) |
| `apply_log1p()` | wendet log1p() auf ausgewählte Spalten an |
| `main()` | orchestriert den gesamten Prozess |

---

## 5. Beispielausgabe (gekürzt)
[INFO] Splits gefunden: 57 (0..56)
[INFO] Features (74): Price, Volume, DivYld12m, MktCap, Return_raw, Return_lag1, ...
[INFO] log1p auf: Price, Volume, MktCap, Return_STD10, ...
[INFO] NaN-Imputation gesetzt auf: 0.0

[OK] Split 000 → train:13930 test:380 | saved: scaled_train_000.parquet, scaled_test_000.parquet, scaler_stats_000.parquet
...
[DONE] Scaling fertig.

---

## 6. Ergebnisse

Nach erfolgreichem Lauf:
- 57 Splits × (Train/Test/Stats/Meta) → ca. 200 Dateien  
- alle standardisiert und reproduzierbar  
- Grundlage für nachfolgende Modellierungsphase (`train_model_splits.py`)

Der Datensatz ist jetzt **modellbereit** – jedes ML-Modell kann trainiert werden,  
ohne dass ein Blick in zukünftige Daten erfolgt.

---

## 7. Warum dieser Aufwand?

In Zeitreihendaten (v. a. Finanzdaten) ist **Leckage subtil**.  
Schon eine falsch berechnete Standardabweichung kann ein Modell „wissen lassen“,  
was im nächsten Monat passiert.  

Durch diesen Prozess:
- werden alle Feature-Skalen vergleichbar,  
- jede Statistik stammt **ausschliesslich aus der Vergangenheit**,  
- spätere Backtests und Modellvergleiche werden **methodisch fair**.

---

## 8. Kritische Betrachtung

**Stärken:**
- saubere Trennung von Train/Test, keine Zukunftsinformation  
- skalierte, reproduzierbare Basis für ML-Training  
- flexibel: log1p, NA-Imputation, Feature-Auswahl  
- effizient trotz 57 Splits (~1–2 min Laufzeit)

**Schwächen / Grenzen:**
- Feature-Selektion erfolgt rein numerisch – keine semantische Prüfung  
- log1p-Heuristik (Schiefe > 1) kann über- oder untertransformieren  
- keine Persistenz der Scaler-Objekte im sklearn-Format (nur Werte)  
- kein Pipeline- oder Parallelisierungsmodus (alles sequentiell)

**Gedanke:**  
Das Skript priorisiert methodische Sauberkeit über Rechenzeit.  
In einer produktiven Umgebung würde man Teile davon parallelisieren  
oder in einer ML-Pipeline kapseln.

---

## 9. Nächste Schritte (To-Dos)

1. **Modellphase starten:**  
   Trainiere erste Modelle (`RandomForest`, `LightGBM`, `XGBoost`) pro Split  
   → Performance messen (IC, Precision@K, Sharpe).

2. **Feature-Stabilität analysieren:**  
   Prüfe, welche Features über alle Splits hinweg stabil hohe Korrelation zum Ziel zeigen.

3. **Scaler-Validierung:**  
   Vergleiche Mittelwerte & Standardabweichungen pro Ticker über Zeit –  
   große Drift könnte auf Strukturbrüche hinweisen.

4. **Pipeline-Automatisierung:**  
   Bias-Review → Split → Scaling → Model → Backtest → Report  
   → zusammenführen in eine konsistente ML-Workflow-Chain.

5. **Parallelisierung (optional):**  
   Splits mit `joblib` oder `multiprocessing` parallel verarbeiten, um Laufzeit zu reduzieren.

---

## 10. Fazit

`scale_and_save.py` ist der Brückenschritt zwischen Rohdaten und Modelltraining.  
Es schafft ein **statistisch sauberes Fundament**:  
jedes Modell trainiert auf vergangene, korrekt normalisierte Werte  
und validiert auf strikt unbekannte, gleichbehandelte Zukunftsdaten.  

Damit ist der gesamte AMC-Datensatz **modellfertig**,  
und der nächste logische Schritt ist die **Evaluierung der Handelslogik im Backtest**.
