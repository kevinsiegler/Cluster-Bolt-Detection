# scripts/infer.py
from ultralytics import YOLO
import os
from datetime import datetime

# Pfad zu deinem trainierten Modell
TRAIN_NAME = "train_30_epoch" #Trainingsdatei einfügen ###############################################################
MODEL_PATH = f"runs/detect/training/{TRAIN_NAME}/weights/best.pt"

# Quelle: Ordner mit Bildern oder einzelnes Bild
IMG = "../dataset/images/val"

# Benutzerdefinierter Name für diese Evaluierung
OUTPUT_NAME = "infer_train_30_epoch_all_confidence"  # <--- deinen Namen setzen #####################################################

# Basispfad für alle Evaluierungen
EVAL_BASE = "runs/detect/evaluations"

# Sicherstellen, dass der übergeordnete Evaluations-Ordner existiert
os.makedirs(EVAL_BASE, exist_ok=True)

# Vollständiger Zielpfad für diese Auswertung
OUTPUT_DIR = os.path.join(EVAL_BASE, OUTPUT_NAME)

# Modell laden
model = YOLO(MODEL_PATH)


# Val-Modus erzeugt JSON
metrics = model.val(
    data="../dataset/data.yaml",  # Dataset YAML
    conf=0.05,
    imgsz=1024,
    save_json=True,
    project="runs/detect/evaluations",
    name=OUTPUT_NAME,
    exist_ok=True
)
