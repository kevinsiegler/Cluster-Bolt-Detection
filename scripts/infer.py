# scripts/infer.py
from ultralytics import YOLO
import os
from datetime import datetime

# Pfad zu deinem trainierten Modell
TRAIN_NAME = "train_30_epoch_(maybe)" #Trainingsdatei einfügen ###############################################################
MODEL_PATH = f"runs/detect/training/{TRAIN_NAME}/weights/best.pt"

# Quelle: Ordner mit Bildern oder einzelnes Bild
IMG = "../dataset/images/val"

# Benutzerdefinierter Name für diese Evaluierung
OUTPUT_NAME = "infer_train_30_epoch_(maybe)"  # <--- deinen Namen setzen #####################################################

# Basispfad für alle Evaluierungen
EVAL_BASE = "runs/detect/evaluations"

# Sicherstellen, dass der übergeordnete Evaluations-Ordner existiert
os.makedirs(EVAL_BASE, exist_ok=True)

# Vollständiger Zielpfad für diese Auswertung
OUTPUT_DIR = os.path.join(EVAL_BASE, OUTPUT_NAME)

# Modell laden
model = YOLO(MODEL_PATH)

# Inferenz starten und in den Evaluations-Ordner speichern
results = model.predict(
    source=IMG,
    conf=0.25,
    imgsz=1024,
    save=True,
    save_txt=True,
    project=EVAL_BASE,     # Basis: runs/detect/evaluations
    name=OUTPUT_NAME,      # dein Ordnername
    exist_ok=False         # falls schon vorhanden, macht automatisch _2, _3 etc.
)

