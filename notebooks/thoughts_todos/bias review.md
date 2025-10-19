# Bias Review & Reflections  
**Datum:** 2025-10-19  
**Autor:** Fabia Holzer  

---

## 1. Beobachtungen aus Bias Check v1  

- **Ticker-Abdeckung:** 20 SMI-Titel, gleichverteilt, keine Überrepräsentation.  
- **Zeitliche Abdeckung:**  
  - Durchschnittlich ca. **10 Jahre Historie** (2015-09-30 bis 2025-09-30).  
  - Abweichungen bei **ALC** (Kotierung 2019) und **AMRZ** (Kotierung 2020).  
  - Diese beiden reduzieren den *gemeinsamen Zeitbereich* auf rund **4–6 Jahre**.  
- **Preisverfügbarkeit:** Alle Berechnungen basieren auf Tagen mit gültigem Preiswert (> 0). Keine künstliche Verlängerung mehr durch leere Zeilen.  
- **IPO-Effekt:** Späte Eintritte führen zu ungleicher Datenlänge je Ticker. Kein struktureller Bias, aber potenziell Einfluss auf Feature-Stabilität bei gleitenden Fenstern.  

---

## 2. Erste analytische Gedanken  

### Data Completeness Bias  
Ticker mit kürzerer Historie haben weniger Trainingsbeispiele → geringere statistische Signifikanz einzelner Features.  
Für Machine-Learning-Modelle könnte das zu verzerrten Gewichtungen führen, wenn alle Ticker gleich behandelt werden.  

### Temporal Alignment  
Um faire Zeitfenster zu schaffen, sollten Modelltrainings ab einem Zeitpunkt beginnen, an dem alle aktiven Titel im Sample enthalten sind (z. B. ab 2021).  
Alternativ: Features per Index normalisieren, um IPO-Titel nicht auszuschliessen, sondern vergleichbar zu skalieren.  

### Survivorship Bias  
Da nur heute aktive SMI-Mitglieder im Datensatz stehen, fehlen ehemalige Index-Mitglieder (z. B. CSGN).  
Das kann Performance-Schätzungen systematisch nach oben treiben.  
Eine spätere Erweiterung um historische SMI-Komponenten wäre sinnvoll.  

### Liquidity Bias  
SMI-Titel sind zwar hochliquide, aber bei Erweiterung auf Nebenwerte müssen Volumen-Filter hinzukommen,  
um Verzerrungen durch illiquide Titel zu vermeiden.  

---

## 3. Empfehlungen & To-Dos  

1. **Trainingsfenster prüfen**  
   - Gemeinsame Zeitbasis ab 2021 definieren.  
   - Alternativ separate Modelle pro Subperiode trainieren.  

2. **Feature-Engineering anpassen**  
   - Fehlende Preisdaten bei IPOs *nicht* durch Interpolation auffüllen.  
   - Stattdessen Nullwerte als *nicht handelbare Tage* kennzeichnen.  

3. **Weighting überdenken**  
   - Gleichgewichtung aller Ticker im Loss-Function-Setup kann Kurz-Historien überbetonen.  
   - Option: Sample-Weights nach Anzahl gültiger Beobachtungen.  

4. **Reporting erweitern**  
   - IPO-Ticker automatisch markieren.  
   - Durchschnittliche Dauer und Median-Beobachtungszahl pro Ticker ergänzen.  

---

## 4. Weiterführende Überlegungen  

> „Bias entsteht selten aus Daten, meist aus impliziten Annahmen.“  
> — Darum: Annahmen sichtbar machen, bevor Modelle sie übernehmen.  

- Die aktuelle Version bildet ein **sauberes SMI-Baseline-Set** – ideal für initiale Backtests.  
- Für die AMC-Implementierung ist wichtig, dass kein struktureller Vorteil einzelner Ticker entsteht (z. B. längere Historie = mehr Einfluss).  
- In einer späteren Iteration könnten **rollierende Trainingsfenster** eingeführt werden, die IPO-Titel schrittweise integrieren, sobald genügend Historie vorhanden ist.  

---

## 5. Fazit  

Der Bias-Check bestätigt, dass die SMI-Datenbasis solide und gleichmässig ist.  
Die Hauptverzerrung entsteht durch IPO-Zeiten – technisch sauber, aber methodisch relevant.  
Diese Erkenntnisse sollten direkt in die Definition der Trainingsperiode und das Feature-Weighting einfliessen.  

---
