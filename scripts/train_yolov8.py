# scripts/train_yolov8.py
from ultralytics import YOLO
import os

# ============================================================
# KONFIGURATION
# ============================================================
MODEL = "yolov8n.pt"   # oder yolov8m.pt für genauere Erkennung
DATA = "../dataset/data.yaml"
IMG_SIZE = 1024
EPOCHS = 1
BATCH = 8
DEVICE = "cpu"
TRAIN_NAME = "train_one_epoch_v1"   # beliebiger Name für das Training

# Ordnerstruktur
PROJECT_DIR = "runs/detect/training"

# Falls der Ordner noch nicht existiert, wird er erstellt
os.makedirs(PROJECT_DIR, exist_ok=True)

# ============================================================
# TRAINING
# ============================================================
model = YOLO(MODEL)
model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    name=TRAIN_NAME,
    project=PROJECT_DIR  # <- sorgt für saubere Struktur
)

print(f"✅ Training abgeschlossen. Ergebnisse in: {os.path.join(PROJECT_DIR, TRAIN_NAME)}")
