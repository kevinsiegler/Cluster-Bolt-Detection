import sys
import os
from ultralytics import YOLO

# =====================
# Modellname aus Argument
# =====================
TRAIN_NAME = sys.argv[1]

# =====================
# Fixer Confidence-Wert
# =====================
CONF = 0.2   # ← hier EINMAL festlegen

# =====================
# Pfade
# =====================
MODEL_PATH = (
    f"runs/detect/train_w_different_amounts_data/"
    f"{TRAIN_NAME}/weights/best.pt"
)

IMG = "../dataset/images/val"

EVAL_BASE = "runs/detect/evaluations_w_different_amounts_data_conf(0.2)"
os.makedirs(EVAL_BASE, exist_ok=True)

OUTPUT_NAME = f"infer_{TRAIN_NAME}"

# =====================
# Sicherheitscheck
# =====================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modell nicht gefunden: {MODEL_PATH}")

print(f"🚀 Starte Inferenz")
print(f"   Modell: {TRAIN_NAME}")
print(f"   Confidence: {CONF}")

# =====================
# YOLO Inferenz
# =====================
model = YOLO(MODEL_PATH)

model.predict(
    source=IMG,
    conf=CONF,
    imgsz=1024,
    save=False,
    save_txt=True,
    save_json=False,
    project=EVAL_BASE,
    name=OUTPUT_NAME,
    exist_ok=False
)

print(f"✅ Inferenz abgeschlossen für {TRAIN_NAME}")
