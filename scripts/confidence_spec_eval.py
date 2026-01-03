from ultralytics import YOLO
import os

# ============================================================
# MANUELLE KONFIGURATION (wie gewünscht)
# ============================================================
TRAIN_NAME = "train_30_epoch"
MODEL_PATH = f"runs/detect/training/{TRAIN_NAME}/weights/best.pt"
DATA = "../dataset/data.yaml"

IMG_SIZE = 1024
DEVICE = "cpu"

# Optional: eigener Name für diese Auswertung
EVAL_NAME = "conf_train_30_epoch_spectrum"

# ============================================================
# AUSWERTUNG
# ============================================================
model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA,
    imgsz=IMG_SIZE,
    conf=0.01,        # ↓ GANZ WICHTIG → Confidence-Spektrum
    iou=0.5,
    device=DEVICE,
    plots=True,       # ← erzeugt PR / P / R / F1 Kurven
    save_json=True,
    project="runs/detect/evaluations",
    name=EVAL_NAME
)