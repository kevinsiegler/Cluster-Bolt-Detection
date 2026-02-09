# GNN-basierter Filter für YOLOv8-Ergebnisse

Dieses Projekt implementiert eine zweite KI-Schicht in Form eines Graph Neural Networks (GNN), um die Ergebnisse eines YOLOv8-Modells zur Schraubenerkennung zu validieren und zu filtern.

## 🎯 Projektziel

Das primäre Ziel ist die Reduzierung von *False Positives* aus dem YOLO-Modell, ohne dabei den *Recall* zu beeinträchtigen. Das GNN analysiert nicht die Bilddaten, sondern ausschließlich die **räumliche Anordnung** der von YOLO erkannten Bounding Boxes. Es lernt das typische Layout von Schrauben auf einem "gesunden" Fahrzeugunterboden und identifiziert Ausreißer, die strukturell unplausibel sind.

## 🛠️ Installation

Bevor Sie die Skripte ausführen, stellen Sie sicher, dass alle Abhängigkeiten korrekt installiert sind. Führen Sie dazu den folgenden Befehl im `scripts/GNN`-Ordner aus:

```bash
pip install -r requirements.txt
```

## � Ordnerstruktur

Die Skripte erwarten die folgende Projektstruktur. Alle Pfade in den Skripten sind relativ zum Hauptverzeichnis des Projekts (`bolt_detection`).

```
bolt_detection/
├── dataset/
│   ├── labels/
│   │   ├── train/      # (Input) Saubere YOLO-Labels für das GNN-Training
│   │   └── val/        # Ground-Truth-Labels für die YOLO-Validierung
│   └── images/
│       ├── train/
│       └── val/
└── scripts/
    ├── GNN/            # Alle Skripte für die GNN-Pipeline
    │   ├── outputs
    │   │   ├── cleaned_labels
    │   │   └── visualizations
    │   ├── trained_models
    │   ├── model.py
    │   ├── utils.py
    │   ├── train_gnn.py
    │   ├── infer_gnn.py
    │   ├── visualize.py
    │   └── README.md   # Diese Datei
    └── YOLO/
        └── ...         # Deine YOLO-Skripte
```

## ⚙️ Workflow

Der Prozess ist in 3+1 Schritte unterteilt:

### Schritt 1: YOLO-Inferenz durchführen

Zuerst benötigen wir die Roh-Ergebnisse von YOLO mit einer sehr niedrigen Konfidenzschwelle, um einen maximalen Recall sicherzustellen. Das Skript `infer_model_with_confidence.py` ist dafür ideal.

- **Input**: Bilder (z.B. aus `dataset/images/val`).
- **Output**: YOLO-Labeldateien (`.txt`) mit 6 Spalten: `class_id x y w h confidence`. Diese müssen im Ordner `scripts/YOLO/infer` liegen.

### Schritt 2: GNN-Modell trainieren

Das GNN wird einmalig auf den "gesunden" Trainingsdaten trainiert, um die normale Schraubenanordnung zu lernen.

- **Input**: Die sauberen Labeldateien aus `dataset/labels/train`.
- **Output**: Ein trainiertes Modell (`gnn_model.pth`) und eine Konfigurationsdatei (`config.json`) im Ordner `scripts/GNN/trained_models`.

**Befehl (aus dem Ordner `scripts/GNN` ausführen):**
```bash
python train_gnn.py --k 5 --epochs 100
```

### Schritt 3: GNN-Inferenz (Filterung)

Das trainierte GNN wird nun auf die rohen YOLO-Ergebnisse aus Schritt 1 angewendet, um unplausible Bounding Boxes zu entfernen.

    1. Der Ordner mit den rohen YOLO-Labels (`scripts/YOLO/infer`).
    2. Das trainierte GNN-Modell aus `gnn_trained_models`.

**Befehl (aus dem Ordner `scripts/GNN` ausführen):**
```bash
python infer_gnn.py --threshold 0.5
```

### Schritt 4: Ergebnisse visualisieren (Optional)

Um die Ergebnisse zu analysieren und zu verstehen, was das GNN tut, kann das Visualisierungsskript verwendet werden. Es erzeugt ein Vergleichsbild (YOLO vs. YOLO+GNN).

- **Input**:
    1. Ein Originalbild.
    2. Die zugehörige **rohe** Labeldatei aus Schritt 1.
    3. Das trainierte GNN-Modell.
- **Output**: Ein Bild in `gnn_outputs/visualizations`, das die Filterung zeigt.

**Befehl (aus dem Ordner `scripts/GNN` ausführen):**
```bash
python visualize.py --image_path ../../dataset/images/val/bild_name.jpg --label_path PFAD_ZUR_ROHEN_LABEL_DATEI.txt --threshold 0.5 --show_edges
```

---

## 📜 Skript-Details

### `train_gnn.py`
- **Zweck**: Trainiert das GNN-Modell.
- **Logik**:
    1. Liest alle `.txt`-Dateien aus dem `--data_path`.
    2. Baut für jede Datei einen Graphen, wobei jede Bounding Box ein Knoten ist.
    3. Trainiert das Modell darauf, alle Knoten als "plausibel" (Label 1.0) zu klassifizieren.
- **Wichtige Parameter**:
    - `--data_path`: Pfad zu den Trainingslabels (Standard: `../../dataset/labels/train`).
    - `--model_dir`: Speicherort für das trainierte Modell (Standard: `../../gnn_trained_models`).
    - `--k`: Anzahl der Nachbarn, die für die Graph-Konstruktion berücksichtigt werden. Ein entscheidender Hyperparameter.
    - `--epochs`: Anzahl der Trainingsdurchläufe.

### `infer_gnn.py`
- **Zweck**: Wendet das trainierte GNN an, um YOLO-Ergebnisse zu filtern.
- **Logik**:
    1. Lädt das trainierte Modell aus `--model_dir`.
    2. Verarbeitet jede `.txt`-Datei aus dem `--input_dir`.
    3. Erstellt pro Datei einen Graphen und lässt das GNN für jeden Knoten (jede Box) eine Plausibilitäts-Wahrscheinlichkeit berechnen.
    4. Behält nur die Boxen, deren Wahrscheinlichkeit über dem `--threshold` liegt.
    5. Speichert die gefilterten Boxen im `--output_dir`.
- **Wichtige Parameter**:
    - `--input_dir`: **(Erforderlich)** Pfad zu den rohen YOLO-Labels, die gefiltert werden sollen.
    - `--output_dir`: Speicherort für die bereinigten Labels (Standard: `../../gnn_outputs/cleaned_labels`).
    - `--model_dir`: Pfad zum trainierten Modell (Standard: `../../gnn_trained_models`).
    - `--threshold`: Der Schwellenwert (0-1) für die Plausibilität. Boxen unter diesem Wert werden als False Positives entfernt. Dies ist der wichtigste Parameter zur Steuerung der Filterstärke.

### `visualize.py`
- **Zweck**: Erstellt eine visuelle Gegenüberstellung der Ergebnisse vor und nach der GNN-Filterung.
- **Logik**:
    1. Lädt ein Bild und die zugehörige rohe Labeldatei.
    2. Berechnet die GNN-Plausibilitäten für alle Boxen.
    3. Erzeugt ein Bild, das links alle rohen Boxen und rechts die farblich markierten (grün=plausibel, rot=unplausibel) Boxen zeigt.
- **Wichtige Parameter**:
    - `--image_path`: **(Erforderlich)** Pfad zum Originalbild.
    - `--label_path`: **(Erforderlich)** Pfad zur rohen YOLO-Labeldatei.
    - `--output_dir`: Speicherort für das Ausgabebild (Standard: `../../gnn_outputs/visualizations`).
    - `--threshold`: Schwellenwert für die Farbkodierung (grün/rot).
    - `--show_edges`: Wenn gesetzt, werden die Kanten des Graphen im Bild eingezeichnet.

### `model.py`
- **Zweck**: Definiert die Architektur des GNN (`AnomalyGNN`).
- **Struktur**: Besteht aus zwei `SAGEConv`-Layern und einem MLP-Klassifikator. Nimmt die Node-Features entgegen und gibt pro Node einen Logit-Wert für die Plausibilität aus.

### `utils.py`
- **Zweck**: Enthält Hilfsfunktionen, die von mehreren Skripten verwendet werden.
- **Funktionen**:
    - `parse_yolo_labels()`: Liest und parst `.txt`-Dateien im YOLO-Format.
    - `build_graph_from_boxes()`: Konstruiert aus einer Liste von Bounding Boxes ein `torch_geometric.data.Data`-Objekt (einen Graphen) mittels k-Nearest-Neighbors.