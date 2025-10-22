# Rolling / Expanding Window CV – Design, Begründung & Reflexion  
**Datum:** 2025-10-22  
**Autorin:** Fabia Holzer  

---

## 1. Überblick

**Ziel:**  
Dieses Skript wurde entwickelt, um **Trainings- und Testaufteilungen für ein ML-basiertes Aktienmodell** zu erzeugen, die **zeitlich valide und reproduzierbar** sind.  
Es soll verhindern, dass das Modell versehentlich Informationen aus der Zukunft nutzt (Data Leakage) oder durch unfaire Splits verzerrt wird.  

**Kernidee:**  
Anstatt zufälliger Folds arbeitet das Skript mit **rollierenden Zeitfenstern** – jedes Fenster trainiert auf Vergangenheit und testet auf Zukunft.  
Das ist die einzige Methode, die einer realen Handelssituation entspricht:  
Man trainiert auf historische Daten und prüft, wie gut das Modell auf neue, noch unbekannte Daten reagiert.  

---

## 2. Warum Rolling / Expanding Window?

### Problem bei klassischen Methoden:
- Zufälliges Mischen von Zeitreihen verletzt die zeitliche Kausalität.  
- Modelle lernen Muster, die in der Praxis **niemals im Voraus verfügbar** wären.  
- Ohne Embargo überlappen sich Training und Test oft (z. B. durch verzögerte Marktreaktionen).  

### Lösung durch Rolling Windows:
- Das Modell trainiert auf einem festen Zeitraum (z. B. 3 Jahre) und wird dann auf den **nächsten Monat** getestet.  
- Nach jedem Durchlauf wird das Fenster ein Stück nach vorne verschoben → realistische Simulation eines fortlaufenden Trainings.  
- Ein **Embargo von 5 Tagen** verhindert, dass Daten zu nahe beieinanderliegen (z. B. Gewinneffekte, Nachlaufeffekte).  

### Warum nicht Expanding Window?  
Rolling und Expanding werden beide unterstützt.  
- Rolling = konstanter Fokus, gut für Modelle mit begrenztem Gedächtnis.  
- Expanding = wachsendes Wissen, aber höheres Risiko von Drift.  
Für AMC-Modelle wurde Rolling gewählt, um Marktregime getrennt zu halten.

---

## 3. Cold-Start-Policy – Wieso nötig?

Nicht jeder Titel hat gleich lange Historie.  
Neue IPOs (z. B. **ALC** 2019, **AMRZ** 2025) verzerren die Statistik, wenn sie zu früh einbezogen werden.  

**Begründung:**  
Ein Modell, das nur wenige Dutzend Datenpunkte zu einem Ticker hat, kann keine stabilen Muster erkennen.  
Diese „kalten“ Titel führen zu schwankenden Splits und erhöhen die Varianz.  

**Lösung:**  
Ticker werden erst berücksichtigt, wenn sie im Trainingsfenster **mindestens 250 Tage** beobachtet wurden.  
Das ist ein pragmatischer Kompromiss: genug Daten, um Verhalten zu lernen – aber keine künstliche Ausschliessung.  

---

## 4. Embargo & Purging – Wieso 5 Tage?

In Finanzdaten wirken viele Ereignisse verzögert.  
Wenn man direkt angrenzende Tage zwischen Train und Test verwendet, kann ein Modell indirekt **zukünftige Information** nutzen –  
z. B. durch überlappende Rückberechnungen, glättende Fenster oder Korrelationen über Feiertage.  

Ein **Embargo von 5 Handelstagen** ist ein Erfahrungswert aus der Literatur (Lopez de Prado, *Advances in Financial Machine Learning*).  
Es eliminiert den größten Teil der ungewollten Signalübertragung, ohne die Stichprobengrösse stark zu verringern.

---

## 5. Technische Struktur

| Komponente | Aufgabe | Warum diese Designentscheidung |
|-------------|----------|--------------------------------|
| `detect_ticker_column()` | Automatische Spaltenerkennung für Ticker. | Macht das Skript robust gegenüber wechselnden Datenquellen. |
| `make_splits()` | Rolling/Expanding-Split mit Embargo. | Vollständig zeitreihensicher, ohne Randomisierung. |
| `Split.meta()` | Liefert Metriken und Verteilungen pro Split. | Transparenz über Datenbalance und Tickerabdeckung. |
| `write_log()` | Speichert Split-Infos als CSV. | Dient der späteren Bias-Analyse und Dokumentation. |
| `save_split_indices()` | Exportiert Indizes (`.pkl`) für Reproduzierbarkeit. | Garantiert identische Splits für Scaling, Training und Backtesting. |

**Warum `.pkl` statt `.csv`?**  
Pickle-Dateien speichern Python-Objekte (hier: Integer-Indizes) verlustfrei.  
Sie sind kompakt, direkt in Pandas einlesbar und vermeiden Rundungsfehler oder Typkonvertierungen.

---

## 6. Beispielergebnis (Interpretation)

**Beispiel-Split:**  
- Train: 2017-12-27 → 2020-12-27  
- Test: 2021-01-01 → 2021-02-01  
- Embargo: 5 Tage  
- Train-Rows: 13’930, Test-Rows: 380  
- Ø Ticker/Tag: 18.7  
- Splits gesamt: 57  

**Gedanke:**  
Diese Struktur imitiert den Ablauf eines echten Fondsmanagements:  
Alle 14 Tage oder 1 Monat wird neu evaluiert, auf Basis der letzten drei Jahre Marktgeschichte.  
Das schafft die Grundlage für eine stabile, inkrementelle Modellbewertung.

---

## 7. Warum Logging & Index-Export?

In Finanzprojekten müssen Ergebnisse **nachvollziehbar und prüfbar** sein – regulatorisch wie analytisch.  
Deshalb wird jeder Split als Indexliste gespeichert:



So kann jedes nachfolgende Modul (z. B. Scaling, ML-Modelle, Backtest) exakt dieselben Daten verwenden.  
Wenn ein Modell später verbessert wird, bleibt der Vergleich **methodisch fair**.

---

## 8. Kritische Reflexion

**Stärken:**
- Zeitlich saubere Trennung → kein Data Leakage.  
- Reproduzierbare Struktur → Backtests werden vergleichbar.  
- Cold-Start-Filterung → realistischere Modellbasis.  
- Modularität → einfach in weitere Pipelines integrierbar.

**Schwächen:**
- Späte IPOs (z. B. AMRZ) führen zu stark schrumpfenden Datensätzen.  
- Keine Gewichtung nach Liquidität oder Handelsvolumen.  
- Rolling-Fenster können driften, wenn Makroregime stark wechseln.  
- Splits sind fix → keine adaptive Optimierung nach Marktvolatilität.

**Gedanke:**  
Das Skript ist bewusst konservativ – lieber weniger, aber saubere Daten,  
als hohe Varianz durch fragwürdige „historische“ Datenpunkte.  
Diese Strenge bildet das Fundament für glaubwürdige Performanceanalysen.

---

## 9. Nächste Schritte (To-Dos)

1. **Feature-Scaling (log1p + z-Score):**  
   Aufbau eines Moduls `scale_and_save.py`, das pro Split und Ticker normalisiert  
   (Train-Stats → Test übernehmen).

2. **Backtesting-Logik:**  
   Entwicklung einer Simulationsumgebung, die die Splits nutzt,  
   um Handelsregeln (Ranking, Gewichtung, Haltedauer, Drawdown-Limits) zu testen.

3. **Feature-Drift-Analyse:**  
   Ermittlung, welche Features über Zeit an Stabilität verlieren  
   → wichtig für langfristig robuste Modelle.

4. **Performance-Monitoring:**  
   Sammlung der Split-Ergebnisse zu globalen Kennzahlen:  
   Information Coefficient, Precision@K, Sharpe Ratio, Max Drawdown.

5. **Automatisierung:**  
   Integration aller Schritte (Bias Review → Splits → Scaling → Model → Backtest → Report)  
   in eine einheitliche Pipeline für kontinuierliche Modellbewertung.

---

## 10. Fazit

Dieses Skript ist die **methodische Basis** der gesamten AMC-Modellphase.  
Es trennt sauber Vergangenheit und Zukunft, erlaubt reproduzierbare Experimente  
und schafft damit die Grundlage für statistisch belastbare Investmententscheidungen.

Kurz gesagt:  
👉 **Ohne sauberes Splitting ist jede Backtest-Performance wertlos.**  
Dieses Skript sorgt dafür, dass das Fundament stimmt – bevor das Modell bewertet wird.
