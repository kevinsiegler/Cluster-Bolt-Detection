# GNN Spatial Validator – README

## Ziel des Systems

Dieses System dient als **räumliche Validierungsschicht für einen Kandidaten-Generator** (z.B. das Cluster-Modul), um die Präzision bei der Vervollständigung von Schraubenmustern drastisch zu erhöhen.

Ziel ist es, geometrisch unplausible Kandidaten für fehlende Schrauben (Falsch-Positive) zu identifizieren und zu entfernen, um eine hohe Präzision zu erreichen, während der hohe Recall des vorgeschalteten Systems erhalten bleibt.

Der GNN-Validator prüft ausschließlich die Frage:

> „Ist dieser Kandidat für eine fehlende Schraube geometrisch plausibel im lokalen Kontext der bereits vorhandenen Schrauben?“

Das System basiert auf einem **Graph-Autoencoder**, der auf die normalen, räumlichen Anordnungen von Schrauben trainiert wird.

---

## Architektur & Funktionsweise

Die Pipeline ist in drei Hauptphasen unterteilt: Training, Inferenz und Visualisierung.

**1. Training (Lernen der "normalen" Geometrie):**

`Idealisierte Prototypen (aus Cluster-Modell)` → `Graph-Erstellung (k-NN)` → `Training des Graph-Autoencoders`

- Aus den Ground-Truth-Labels (nur Positionen, keine Klassen) werden für jedes Bild Graphen erstellt.
- Jede Bounding Box ist ein **Knoten (Node)** im Graph.
- Die **Kanten (Edges)** verbinden jeden Knoten mit seinen `k` nächsten Nachbarn (k-NN).
- Ein Graph-Autoencoder wird trainiert, die Knoten-Features (x, y, w, h) jedes Graphen zu rekonstruieren.
- Das Modell lernt so, wie eine "typische" lokale Schraubenanordnung aussieht.

**2. Inferenz (Validierung der YOLO-Ergebnisse):**

`Vervollständigte Labels (Originale: Klasse 0, Kandidaten: Klasse 1)` → `Graph-Erstellung` → `GNN-Inferenz` → `Anomalie-Prüfung` → `Gefilterte Labels`

- Eine vervollständigte Label-Datei wird geladen. Sie enthält originale Schrauben (Klasse 0) und vom Cluster-Modul hinzugefügte Kandidaten für fehlende Schrauben (Klasse 1).
- Eine Prüfung wird **nur für die Kandidaten (Klasse 1)** durchgeführt:
  1. Ein Graph wird aus **allen** Boxen des Bildes (Originale + Kandidaten) erstellt, um den vollen räumlichen Kontext zu nutzen.
  2. Der trainierte Autoencoder versucht, die Positionen aller Knoten zu rekonstruieren.
  3. Für jeden **Kandidaten-Knoten** wird der **Rekonstruktionsfehler** (Abstand zwischen Original- und rekonstruierter Position) berechnet.
  4. Ist der Fehler größer als der `anomaly_threshold`, wird der Kandidat als geometrische Anomalie (False Positive) markiert und **verworfen**.
  5. Alle originalen Schrauben (Klasse 0) und alle validierten Kandidaten bleiben erhalten.
- Das Ergebnis ist eine bereinigte Label-Datei mit höherer Präzision.

---

## Projektstruktur
...

---

## Ausführung

### Schritt 1: Trainingsdatensatz erstellen
Dieses Skript liest die Ground-Truth-Labels und erstellt die Graphen für das Training.
```bash
python dataset_builder.py
```

### Schritt 2: GNN-Modell trainieren
Trainiert den Graph-Autoencoder mit den zuvor erstellten Graphen.
```bash
python train_gnn.py
```

### Schritt 3: YOLO-Ergebnisse validieren
Lädt die YOLO-Predictions, wendet die GNN-Validierung an und speichert die bereinigten Labels.
```bash
python inference_gnn.py
```

### Schritt 4 (Optional): Ergebnisse visualisieren
Zeigt ein Bild mit den ursprünglichen und den gefilterten Bounding Boxes an.
```bash
python visualize_results.py --image_id <ID_des_Bildes>
```

---

## Wichtige Parameter in `config.yaml`

- `gnn.k_neighbors`: Anzahl der Nachbarn für die k-NN-Grapherstellung. Ein höherer Wert erfasst einen größeren lokalen Kontext. (Empfehlung: 5-10)
- `inference.anomaly_threshold`: Schwellenwert für den Rekonstruktionsfehler. Kandidaten-Boxen mit einem höheren Fehler werden als Anomalie entfernt. Dieser Wert muss experimentell ermittelt werden, ein guter Startpunkt ist ein Wert in der Größenordnung der Bounding-Box-Dimensionen (z.B. 0.015).
