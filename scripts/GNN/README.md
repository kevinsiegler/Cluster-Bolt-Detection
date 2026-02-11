# GNN Spatial Validator – README

## Ziel des Systems

Dieses System dient als **räumliche Validierungsschicht nach YOLOv8**, um die Präzision der Schraubendetektion zu erhöhen.

Ziel ist es, seltene Falsch-Positive-Detektionen (geometrisch unplausible Schrauben-Boxen) zu entfernen, ohne den Recall signifikant zu verschlechtern. Die eigentliche Klassifizierung (Schraube vorhanden/fehlend) wird von YOLOv8 übernommen.

Der GNN-Validator prüft ausschließlich die Frage:

> „Ist diese von YOLO erkannte Schraube (oder Lücke) geometrisch plausibel im lokalen Kontext ihrer Nachbarn?“

Das System basiert auf einem **Graph-Autoencoder**, der auf die normalen, räumlichen Anordnungen von Schrauben trainiert wird.

---

## Architektur & Funktionsweise

Die Pipeline ist in drei Hauptphasen unterteilt: Training, Inferenz und Visualisierung.

**1. Training (Lernen der "normalen" Geometrie):**

`Ground Truth Labels` → `Graph-Erstellung (k-NN)` → `Training des Graph-Autoencoders`

- Aus den Ground-Truth-Labels (nur Positionen, keine Klassen) werden für jedes Bild Graphen erstellt.
- Jede Bounding Box ist ein **Knoten (Node)** im Graph.
- Die **Kanten (Edges)** verbinden jeden Knoten mit seinen `k` nächsten Nachbarn (k-NN).
- Ein Graph-Autoencoder wird trainiert, die Knoten-Features (x, y, w, h) jedes Graphen zu rekonstruieren.
- Das Modell lernt so, wie eine "typische" lokale Schraubenanordnung aussieht.

**2. Inferenz (Validierung der YOLO-Ergebnisse):**

`YOLO-Prediction` → `Confidence-Filter` → `Graph-Erstellung` → `GNN-Inferenz` → `Anomalie-Prüfung` → `Gefilterte Labels`

- Eine YOLO-Prediction wird geladen (`class, x, y, w, h, conf`).
- Boxen mit `confidence >= yolo_confidence_threshold` werden **direkt als korrekt übernommen**.
- Für die restlichen Boxen (`confidence < yolo_confidence_threshold`) wird eine Prüfung durchgeführt:
  1. Ein Graph wird aus **allen** Boxen des Bildes erstellt, um den vollen Kontext zu nutzen.
  2. Der trainierte Autoencoder versucht, die Features aller Knoten zu rekonstruieren.
  3. Für jede niedrig-konfidente Box wird der **Rekonstruktionsfehler** berechnet.
  4. Ist der Fehler größer als der `anomaly_threshold`, wird die Box als Anomalie (False Positive) markiert und **verworfen**.
- Das Ergebnis ist eine bereinigte Label-Datei mit höherer Präzision.

---

## Projektstruktur

```
scripts/
└─ GNN/
   ├─ config.yaml                # Alle Konfigurationen und Pfade
   ├─ requirements.txt
   ├─ utils.py                   # Hilfsfunktionen (Label-Handling, Graph-Erstellung)
   ├─ dataset_builder.py         # Erstellt die Trainings-Graphen aus Ground Truth Labels
   ├─ train_gnn.py               # Trainiert das Graph-Autoencoder-Modell
   ├─ inference_gnn.py           # Validiert YOLO-Predictions und entfernt Anomalien
   ├─ visualize_results.py       # Visualisiert die gefilterten Ergebnisse
   └─ outputs/
       ├─ model.pt                 # Das trainierte Modell
       ├─ datasets/                # Gespeicherte Graphen für das Training
       └─ validated_labels/        # Die bereinigten Label-Dateien
```

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
- `inference.yolo_confidence_threshold`: YOLO-Detektionen über diesem Schwellenwert werden als korrekt angenommen und nicht geprüft. (Empfehlung: 0.1 - 0.5)
- `inference.anomaly_threshold`: Schwellenwert für den Rekonstruktionsfehler. Boxen mit einem höheren Fehler werden als Anomalie entfernt. (Muss experimentell ermittelt werden, z.B. 0.01 - 0.05)
