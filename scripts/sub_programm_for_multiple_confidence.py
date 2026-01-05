# scripts/confidence_analysis.py
import sys
import os
from ultralytics import YOLO

# =====================
# Confidence aus Argument
# =====================
conf = float(sys.argv[1])

# Pfad zu deinem trainierten Modell
TRAIN_NAME = "train_30_epoch_with_parameter_n" 
MODEL_PATH = f"runs/detect/training/{TRAIN_NAME}/weights/best.pt"

# Quelle
IMG = "../dataset/images/val"

# Basispfad für alle Evaluierungen
EVAL_BASE = "runs/detect/evaluations_30_w_param_n"
os.makedirs(EVAL_BASE, exist_ok=True)

# Dynamischer Output-Name
OUTPUT_NAME = f"infer_{TRAIN_NAME}_conf({conf})"

print(f"Starte Inferenz mit conf={conf}")

# Modell laden
model = YOLO(MODEL_PATH)

# Inferenz starten
model.predict(
    source=IMG,
    conf=conf,
    imgsz=1024,        # bleibt wie gewünscht
    #batch=1,            Peak-Schutz (empfohlen)
    save=False,
    save_txt=True,
    save_json=False,
    project=EVAL_BASE,
    name=OUTPUT_NAME,
    exist_ok=False
)

print(f"✅ Inferenz abgeschlossen: conf={conf}")
