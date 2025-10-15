

## Erste Analyse: Top-30-Korrelationen (Makro-Features)

Zur Evaluierung der Eingangsvariablen wurde eine Korrelationsmatrix der 30 am stärksten miteinander verbundenen Makro-Features erstellt.  
Ziel war es, erste Strukturen, Abhängigkeiten und mögliche Redundanzen in den Daten zu identifizieren.

### Beobachtungen

- **Zinsstruktur-Cluster**  
  Zwischenlaufzeiten und Renditen (z. B. `GSWISS10`, `GSWISS20`, `GDBR10`, `USGG10YR`) weisen extrem hohe positive Korrelationen auf.  
  Diese Variablen reagieren auf denselben makroökonomischen Zyklus.  
  Für das Modell reicht voraussichtlich eine repräsentative Laufzeit pro Region, um Informationsüberschneidung zu vermeiden.

- **Makroökonomische Indikatoren**  
  Reihen wie `ECCPEMUY`, `CPURNSA` und `SZCPIYOY` zeigen negative Korrelationen zu den Zinsvariablen – typisch für Phasen steigender Zinsen bei gleichzeitig abnehmender Konjunkturdynamik.  
  Diese Beziehungen sind wirtschaftlich plausibel und bestätigen, dass die Daten interne Konsistenz besitzen.

- **Volatilitäts- und Währungsindizes**  
  Indikatoren wie `VIX`, `VSMIX`, `MOVE`, `USDCHF` oder `XAU` verhalten sich weitgehend unabhängig von den Zins- und Konjunkturblöcken.  
  Sie könnten daher als Regime- oder Risiko-Erklärvariablen dienen und zusätzliche Informationsdimensionen ins Modell bringen.

### Interpretation

Die Heatmap verdeutlicht eine klare Blockstruktur zwischen Zins-, Makro- und Risiko-Faktoren.  
Für das Feature-Engineering bedeutet das:

- Multikollinearität zwischen hoch korrelierten Variablen muss reduziert werden (Feature-Selektion oder PCA).  
- Unabhängige Variablen wie Volatilität oder Währungen sollten erhalten bleiben, da sie die Modellrobustheit erhöhen.  
- Negative Zusammenhänge zwischen Zinsen und Konjunkturindikatoren spiegeln reale Marktmechanismen wider und stützen die Modelllogik.

Diese erste Korrelationsanalyse bildet die Grundlage für die spätere Feature-Auswahl und Validierung der AMC-Modellarchitektur.
![Top 30 Korrelationsmatrix](Rplot_top30_correlation.png)

## Ökonomische Interpretation der Korrelationen

Die Korrelationen innerhalb der Makro-Features zeigen ein konsistentes wirtschaftliches Muster und spiegeln die zugrunde liegende Dynamik globaler Märkte wider.  
Im Datensatz lassen sich drei klar unterscheidbare Strukturebenen erkennen.

### 1. Zinswelt – Synchronisierte Anleihemärkte
Variablen wie `GSWISS10`, `GSWISS20`, `GDBR10` oder `USGG10YR` sind nahezu perfekt positiv korreliert.  
Das verdeutlicht die enge Kopplung der globalen Zinsmärkte: Bewegungen der US-Treasuries übertragen sich unmittelbar auf Schweizer und europäische Staatsanleihen.  
Für das Modell bedeutet das, dass **eine einzelne repräsentative Laufzeit pro Region** meist genügt, da der Informationsgehalt der restlichen hochredundant ist.

### 2. Makrowelt – Realwirtschaftliche Dynamik
Indikatoren wie `CPURNSA`, `ECCPEMUY` und `SZCPIYOY` zeigen eine deutlich **negative Korrelation** zu den Zinsvariablen.  
Ökonomisch ist das nachvollziehbar: Steigende Zinsen kühlen über Kreditkosten und Investitionsdruck die reale Wirtschaft ab, woraufhin Beschäftigung und Preiswachstum nachlassen.  
Diese gegenläufige Bewegung ist typisch für fortgeschrittene Zinszyklen und kann im Modell helfen, **konjunkturelle Regimewechsel** zu erkennen.

### 3. Risikowelt – Marktstimmung und Stress
Volatilitäts- und Risikoindikatoren wie `VIX`, `MOVE` und `VSMIX` bilden weitgehend unabhängige Inseln innerhalb der Matrix.  
Sie reagieren nicht linear auf Zinsen oder Makrodaten, sondern auf Marktunsicherheit.  
Ihre relative Unabhängigkeit macht sie wertvoll als **Frühwarn- oder Regimeindikatoren** in der Portfolio-Logik.

### 4. Währungen und Rohstoffe – Sicherheits- und Risikoassets
- **Gold (`XAU`)** zeigt negative Korrelation zu Zinsen: klassischer *Safe-Haven*-Effekt.  
- **Bitcoin (`XBTUSD`)** korreliert leicht positiv mit Risikoassets und spiegelt spekulatives Sentiment wider.  
- **USD/CHF** bewegt sich zwischen diesen Polen – mal Fluchtwährung, mal Carry-Trade.

### Fazit
Die Top-30-Korrelationen bestätigen eine klare makroökonomische Struktur:
- **Zinswelt** → getrieben von Geldpolitik und globalen Zinszyklen.  
- **Makrowelt** → nachlaufende Reaktion der Realwirtschaft.  
- **Risikowelt** → vorlaufende Marktstimmung und Stressindikatoren.  

Diese Trennung zeigt, dass der Datensatz intern konsistent ist und echte wirtschaftliche Zusammenhänge abbildet.  
Damit bildet die Analyse eine fundierte Basis für Feature-Selektion, Regime-Klassifikation und die spätere Integration in das AMC-Modell.


