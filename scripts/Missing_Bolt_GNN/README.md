# Missing Bolt GNN Completion

Dieses Projekt trainiert ein Graph Neural Network (GNN), um fehlende Schraubenpositionen basierend auf lokalen geometrischen Mustern zu ergänzen.

## 1. Ziel des Projekts

Das Ziel ist die Entwicklung einer KI, die **ausschließlich auf Basis von Koordinaten** lernt, wie Schraubenmuster angeordnet sind. Wenn diese KI später eine unvollständige Anordnung von Schrauben sieht, soll sie die Positionen der fehlenden Schrauben präzise ergänzen.

**Wichtige Design-Entscheidungen:**
- **Keine Bilddaten:** Das Modell sieht niemals die echten Bilder, nur die `(x, y)`-Koordinaten der Schrauben.
- **Fokus auf lokale Muster:** Das Modell soll keine globalen Layouts (z.B. "ganze Autotür") auswendig lernen, sondern nur die Beziehungen zwischen benachbarten Schrauben (z.B. "Schrauben bilden hier eine gerade Linie im Abstand von 5cm").
- **Robustheit:** Das Modell muss mit variabler Anzahl von Schrauben (5-30 pro Bild) und unterschiedlichen Bildausschnitten klarkommen.

## 2. Das Grundprinzip: Denoising & Completion

Das Modell wird nach einer Methode trainiert, die man als "Denoising Graph Completion" bezeichnen kann. Die Kernidee ist einfach:

> "Um zu lernen, wie man etwas vervollständigt, nehmen wir ein perfektes Original, machen es künstlich kaputt und trainieren ein Modell darauf, den Originalzustand wiederherzustellen."

### Schritt A: Der Trainingsprozess (`train.py`)

Wenn du `train.py` startest, passiert für jedes Bild im Trainingsdatensatz Folgendes im Hintergrund:

1.  **Laden der "perfekten" Wahrheit:** Das Skript lädt die Label-Datei. Es ignoriert die Klassen (`0 = Bolt`, `1 = Missing_Bolt`) und behandelt **alle** Punkte als die vollständige, korrekte Schraubenanordnung.

2.  **Künstliches "Kaputtmachen" (Augmentation):**
    *   **Maskierung:** Es werden zufällig 10-40% der Punkte aus der "perfekten" Anordnung entfernt. Diese entfernten Punkte sind das, was das Modell später finden soll (positive Kandidaten).
    *   **Positionsrauschen:** Die Koordinaten der verbleibenden, sichtbaren Punkte werden leicht verschoben (Gaussian Noise). Das macht das Modell robuster gegen kleine Ungenauigkeiten von YOLO.
    *   **Negative Kandidaten:** Es werden zufällige Fake-Punkte im Bildbereich erzeugt. Das Modell muss lernen, diese als "hier gehört keine Schraube hin" zu erkennen.

3.  **Graph-Konstruktion (`graph_builder.py`):**
    *   Aus den **sichtbaren** (übrig gebliebenen) Punkten wird ein Graph erstellt.
    *   Jeder Punkt wird zu einem **Knoten (Node)**.
    *   Jeder Knoten wird mit seinen `k` nächsten Nachbarn verbunden (`k-nearest neighbors`). Diese Verbindungen sind die **Kanten (Edges)**.
    *   Die Kanten bekommen Eigenschaften (Features) wie den Abstand und den Vektor (`dx`, `dy`) zwischen den verbundenen Punkten.

4.  **Training des Modells (`model.py`):**
    *   Das GNN analysiert den "kaputten" Graphen und lernt die darin enthaltenen geometrischen Muster.
    *   Anschließend wird das Modell gefragt: "Wie wahrscheinlich ist es, dass an der Position der entfernten (positiven) und der Fake-Punkte (negativen) eine Schraube hingehört?"
    *   **Loss-Berechnung:** Der "Loss" (Fehler) wird berechnet, indem die Vorhersage des Modells mit der Wahrheit verglichen wird. Das Ziel ist, dass das Modell für die echten entfernten Punkte eine hohe Wahrscheinlichkeit und für die Fake-Punkte eine niedrige Wahrscheinlichkeit vorhersagt.
    *   **Lernen:** Der Fehler wird genutzt, um die internen Parameter des Modells anzupassen (`optimizer.step()`), sodass es beim nächsten Mal eine bessere Vorhersage trifft.

Dieser Prozess wird für tausende Bilder über viele Epochen wiederholt, bis das Modell ein sehr gutes Gefühl für typische Schraubenabstände und -muster entwickelt hat.

### Schritt B: Der Inferenzprozess (`inference.py`)

Nach dem Training wird das gespeicherte Modell genutzt, um **echte** fehlende Schrauben zu finden.

1.  **Laden der unvollständigen Daten:** Das Skript lädt eine Label-Datei aus dem Validierungsset und nutzt **nur die vorhandenen Schrauben (Klasse 0)**. Die `Missing_Bolt`-Einträge (Klasse 1) werden ignoriert.

2.  **Graph-Konstruktion:** Aus den vorhandenen Schrauben wird, genau wie im Training, ein Graph gebaut.

3.  **Generierung von Kandidatenpunkten:** Das Modell weiß nicht, wo es suchen soll. Daher wird ein feines Raster (`grid_density`) über den Bereich der vorhandenen Schrauben gelegt. Jeder Punkt in diesem Raster ist ein potenzieller Kandidat für eine fehlende Schraube.

4.  **Vorhersage:** Das trainierte Modell bekommt den Graphen der vorhandenen Schrauben und die Liste aller Raster-Kandidaten. Für jeden einzelnen Kandidatenpunkt berechnet es die Wahrscheinlichkeit, dass an dieser Stelle eine Schraube hingehört.

5.  **Filterung und Auswahl:**
    *   **Probability Threshold:** Nur Kandidaten, deren Wahrscheinlichkeit über einem Schwellenwert (`prob_threshold`) liegt, werden weiter betrachtet.
    *   **Non-Maximum Suppression (NMS):** Oft sagen mehrere nahe beieinander liegende Rasterpunkte "hier ist eine Schraube!". NMS filtert diese Duplikate heraus und behält nur den Punkt mit der höchsten Wahrscheinlichkeit in einer kleinen Umgebung.

6.  **Ergebnis speichern:** Die finalen, als fehlend erkannten Positionen werden als **Klasse 1** in eine neue Label-Datei geschrieben, zusammen mit den ursprünglich vorhandenen Schrauben (Klasse 0).

### Schritt C: Der Evaluationsprozess (`evaluate.py`)

Dieser Schritt prüft, wie gut das Modell bei der Inferenz war.

1.  **Vergleich:** Das Skript lädt die vom `inference.py`-Skript erzeugte Label-Datei und die ursprüngliche Ground-Truth-Label-Datei.
2.  **Matching:** Es vergleicht die Positionen der **vorhergesagten `Klasse 1`-Punkte** mit den Positionen der **echten `Klasse 1`-Punkte**.
3.  **Metriken:** Es berechnet:
    *   **True Positives (TP):** Eine vorhergesagte Schraube war tatsächlich eine fehlende Schraube.
    *   **False Positives (FP):** Eine vorhergesagte Schraube war an einer falschen Stelle.
    *   **False Negatives (FN):** Eine echte fehlende Schraube wurde vom Modell nicht gefunden.
    *   Daraus werden **Precision, Recall und F1-Score** berechnet, um die Gesamtleistung zu bewerten.

## Ordnerstruktur

```text
Missing_Bolt_GNN/
├── config.yaml             <-- ZENTRALE STEUERUNG
├── train.py                <-- Training
├── inference.py            <-- Vorhersage (Ergänzung)
├── evaluate.py             <-- Auswertung der Vorhersage
├── utils.py
├── model.py
├── graph_builder.py
├── data_preparation.py
└── output/                 <-- Automatisch generiert
    ├── training/
    │   └── run_v1/         <-- Gespeichertes Modell & Config-Kopie
    ├── inference/
    │   └── infer_test_01/  <-- Ergänzte Label-Dateien
    └── evaluation/
        └── infer_test_01/  <-- Auswertungs-Reports
```

## 3. Anleitung zur Ausführung

Der gesamte Prozess wird über die `config.yaml` gesteuert.

### Schritt 1: Training

1.  Öffne `config.yaml`.
2.  Unter `training`: Setze einen eindeutigen `run_name` (z.B. `"run_v1_mit_mehr_noise"`).
3.  Führe das Training aus:
    ```bash
    python train.py
    ```
    *Ergebnis:* Das trainierte Modell wird unter `output/training/run_v1_mit_mehr_noise/model.pth` gespeichert.

### Schritt 2: Inferenz (Vorhersage)

1.  Öffne `config.yaml`.
2.  Unter `inference`:
    *   Setze `run_name` für diesen Inferenz-Lauf (z.B. `"infer_mit_modell_v1"`).
    *   Setze `model_train_run` auf den Namen des Trainingslaufs, den du verwenden möchtest (z.B. `"run_v1_mit_mehr_noise"`).
3.  Führe die Inferenz aus:
    ```bash
    python inference.py
    ```
    *Ergebnis:* Die ergänzten Label-Dateien werden in `output/inference/infer_mit_modell_v1/` erstellt.

### Schritt 3: Evaluation

1.  Öffne `config.yaml`.
2.  Unter `evaluation`:
    *   Setze `inference_run` auf den Namen des Inferenzlaufs, den du auswerten willst (z.B. `"infer_mit_modell_v1"`).
3.  Führe die Evaluation aus:
    ```bash
    python evaluate.py
    ```
    *Ergebnis:* Ein detaillierter Bericht wird in `output/evaluation/infer_mit_modell_v1/evaluation_report.txt` gespeichert.

## 4. Wichtige Parameter in `config.yaml`

-   `model.k_neighbors`: Wie viele Nachbarn betrachtet das GNN für jeden Punkt? Ein höherer Wert erfasst größere Muster, kann aber auch zu viel Rauschen aufnehmen. (Standard: 5)
-   `training.mask_ratio_...`: Wie viel Prozent der Punkte werden im Training künstlich entfernt? Dies ist ein entscheidender Wert für die Lernaufgabe.
-   `inference.grid_density`: Wie fein ist das Raster, auf dem nach fehlenden Schrauben gesucht wird? Höher = genauer, aber langsamer.
-   `inference.prob_threshold`: Ab welcher Wahrscheinlichkeit (0-1) wird ein Kandidatenpunkt als "fehlende Schraube" akzeptiert?
-   `evaluation.distance_threshold`: Der maximale Abstand (in normalisierten Koordinaten), den eine vorhergesagte Schraube von der echten haben darf, um als Treffer (True Positive) zu gelten.