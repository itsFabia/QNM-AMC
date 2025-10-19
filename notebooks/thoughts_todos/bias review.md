# Bias Review & Reflections  
**Datum:** 2025-10-19  
**Autor:** Fabia Holzer

## 1. Überblick
- **Zeitraum:** 2015-09-30 bis 2025-09-30  
- **Universe:** 20 SMI-Titel, Verteilung je Ticker 5 %  
- **Empfohlener Trainingsstart (≥ 90 % aktiv):** 2015-09-30  
- **Späte IPOs:** **ALC** (ab 2019-04-09), **AMRZ** (ab 2025-06-23, **71 Beobachtungen**)  
- **Preisfilter:** Berechnungen nur an Tagen mit **Price > 0**

## 2. Zeitliche Abdeckung (Kurzfazit)
- Die meisten Ticker haben ~**10 Jahre** Historie (2515 Beobachtungen).  
- **ALC**: ~6.5 Jahre; **AMRZ**: **0.27 Jahre** → klarer **Cold-Start**.  
- Konsequenz: Gemeinsamer Zeitbereich schrumpft stark, sobald späte IPOs einbezogen werden.

## 3. Feature-Level-Bias (Auszug der markantesten Abweichungen)
**MktCap (global mean 66.2k, std 78.8k):**  
- Hoch: **NESN (+2.60σ)**, **ROG (+2.15σ)**, **NOVN (+1.77σ)**  
- Niedrig: **LOGN (−0.72σ)**, **SLHN (−0.66σ)**, **GEBN (−0.61σ)**  

**DivYld12m (global mean 3.30, std 1.89):**  
- Hoch: **HOLN (+2.09σ)**, **SREN (+1.48σ)**, **ZURN (+0.98σ)**  
- Niedrig: **ALC (−1.60σ)**, **LONN (−1.32σ)**, **SIKA (−1.04σ)**  

**Volume (global mean 1.88 Mio, std 3.47 Mio):**  
- Extrem hoch: **UBSG (+2.77σ)**  
- Hoch: **ABBN (+0.97σ)**, **NESN (+0.78σ)**, **NOVN (+0.72σ)**  
- Sehr niedrig: **PGHN (−0.52σ)**, **SCMN (−0.51σ)**, **SLHN (−0.50σ)**  

**Returns & Volatilität (Beispiele):**  
- **Return_MA/STD**: LOGN, CFR, UBSG tendenziell volatil; **NESN**, **SCMN** niedrig.  
- **Return-Z-Scores (20T):** Mittelwerte nahe 0, leichte negative Drifts; keine offensichtliche systematische Schieflage über alle Titel.

## 4. Risiken und Implikationen
- **Scale-/Level-Bias:** MktCap, Volume, DivYld zeigen starke Levelunterschiede → Modelle können ungewollt „Grösse“ statt Signal lernen.  
- **Horizon-/Data-Completeness-Bias:** **AMRZ** (71 Zeilen) und **ALC** (1629) haben deutlich weniger Evidenz → unsichere Feature-Schätzungen, Overfitting-Risiko.  
- **Rolling-Window-Effekte:** Ungleiche Historien beeinflussen gleitende Statistiken (MA/STD) und deren Stabilität.

## 5. Empfehlungen (umsetzungsnah)
1. **Cross-Validation:** **Rolling/Expanding Window Splits** ohne zufällige Folds; Purge/Embargo passend zum 5-Tage-Horizon.  
2. **Trainingsfenster:** Entweder ab empfohlenem Start (2015-09-30) **oder** ab **2021-01-01** für faire Vergleichbarkeit (späte IPOs).  
3. **Standardisierung bei starken Shifts:**  
   - **log1p** auf **MktCap** und **Volume**;  
   - **Z-Score pro Ticker oder Index**, **ausschliesslich im jeweiligen Train-Split** fitten.  
4. **Späte IPOs separat behandeln:**  
   - **AMRZ** (und ggf. **ALC**) erst scoren, wenn Mindest-Historie erreicht, **oder** separat/hierarchisch modellieren (stark regularisierte Effekte).  
5. **Keine Interpolation von Preisen:** Null/NaN bedeutet „nicht handelbar“, nicht „0“.  
6. **Reporting erweitern:** IPO-Ticker automatisch taggen; Beobachtungszahl & Dauer je Ticker im Fold-Report mitführen.

## 6. To-Dos für die Pipeline
- [ ] Per-Ticker **Scaler/Transformer** (fit nur auf Train), **log1p** für MktCap/Volume.  
- [ ] **Rolling CV** mit Embargo; ein konsistenter **Start ab 2021-01-01** als Baseline.  
- [ ] **Cold-Start-Policy** definieren (z. B. Score erst ab N ≥ 250).  
- [ ] **Fold-Reports**: mean_shift_sigma-Heatmap, Feature-Drift (PSI), OOS-IR & Rank-Spearman je Fold/Ticker.  
- [ ] **Dokumentation**: Preisspalte = `Price` (nur > 0).

## 7. Fazit
Die Datenbasis ist für SMI breit und konsistent; die grösste Verzerrung entsteht durch **Skalenunterschiede** (MktCap/Volume/DivYld) und **sehr kurze Historien** (AMRZ, teils ALC). Beides ist modellseitig handhabbar: **zeitrichtige Standardisierung**, **saubere Rolling-CV** und eine **klare Cold-Start-Policy** verhindern, dass Level oder Datenlänge das Ergebnis dominieren.
