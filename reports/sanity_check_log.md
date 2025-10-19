# Sanity Check Log – AMC_model_input.csv
**Datum:** 2025-10-19  
**Verantwortlich:** Fabia Holzer  

---

## 1. Überblick
**Ziel:** Überprüfung der Datenbasis nach der ersten Aufbereitung über 10 Jahre.  
**Datei:** `AMC_model_input.csv`  
**Zeilen:** 50’300  
**Spalten:** ca. 80 – 90 Features (Markt-, Makro- und Fundamentaldaten)

---

## 2. Struktur-Check
- Keine Duplikate (Date + Ticker): ✅  
- Datentypen stimmen, aber Python warnte beim Einlesen wegen `dayfirst=False`.  
  → Format auf **dd.mm.yyyy** korrigieren (`dayfirst=True`).  
- Datensatz über alle 10 Jahre hinweg konsistent, keine abrupten Sprünge in Spaltenzahl.

---

## 3. Fehlende Werte
**Kritische Lücken bei Makrodaten:**

| Variable | Anteil NaN | Anmerkung |
|-----------|-------------|-----------|
| EURR002W Index | ~64 % | Zinsserie, vermutlich Monats- oder Quartalsdaten. |
| GSWISS10 Index | ~57 % | Gleicher Effekt, abweichende Frequenz. |
| GDBR10 Index | ~32 % | Teilweise unvollständig. |
| SZCPIYOY Index | ~28 % | Inflation, Monatsdaten. |

→ **Bewertung:** Zeitliche Inkonsistenz zwischen Markt- (täglich) und Makro-Reihen.  
**Nächster Schritt:** Forward-Fill + Rolling-Interpolation testen und Bias prüfen.

---

## 4. Ausreisser
- Rund 1 % der Werte > 5 Standardabweichungen.  
- Hauptsächlich Kurs- und Makrodaten während Krisenjahren 2020/2022.  
→ Noch akzeptabel, aber Modell-Robustheit könnte leiden.  
**To-Do:** Winsorizing auf ±3 SD vor dem Modelltraining.

---

## 5. Leak-Check
- Starke Korrelation (0.81) zwischen `Excess_5d_fwd` und `Ret_5d_fwd` – logisch, da Benchmark-Bezug.  
- Alle anderen Features < 0.04 → kein Look-ahead erkennbar.  
→ **Validität:** Gegeben, aber in künftigen Pipelines automatisch prüfen.

---

## 6. Visualisierung (neu)
**Heatmap erfolgreich generiert.**

**Beobachtungen:**
- Deutlich sichtbare Cluster zwischen `Price`, `MktCap`, `Return_lag*` → starke Redundanz.  
- Mehrere Makro-Blöcke (z. B. GDBR10 / GSWISS10) korrelieren untereinander hoch → mögliche Mehrfachzählung derselben Information.  
- Schwache, aber gleichgerichtete Korrelationen zwischen Makro- und Return-Features → Hinweis auf Marktregime-Abhängigkeit.  

**Fazit:**  
Das Feature-Set ist inhaltlich kohärent, aber strukturell zu dicht.  
Feature-Selection oder PCA nötig, um Multikollinearität zu vermeiden.  

---

## 7. Fazit / Einschätzung
- **Positiv:** Keine Duplikate, keine Datenlecks, stabile numerische Struktur.  
- **Kritisch:** Hohe Redundanz zwischen Preis-, Return- und Kapitalisierungsvariablen.  
- **Risiko:** Frequenz-Mismatch und potenzieller Makro-Bias.  
- **Nächster Schritt:** Cleaning + Feature-Reduktion → erneuter Check (V2).

---

## 8. Nächste Schritte
1. **Makro-Imputing:** forward-fill + rolling mean testen.  
2. **Outlier-Handling:** Winsorize ±3 SD.  
3. **Feature-Pruning:** Drop > 0.95 korrelierte Variablen.  
4. **Bias-Review:** prüfen, ob bestimmte Indizes überrepräsentiert sind.  
5. **Dokumentation:** Ergebnisse im Log V2 fortführen, Bias-Notizen aktualisieren.

---

_„Erste Visualisierung hat gezeigt: Die Daten reden – aber sie reden zu viel miteinander.“_
