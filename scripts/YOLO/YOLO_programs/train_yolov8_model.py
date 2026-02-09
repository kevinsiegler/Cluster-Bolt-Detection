# scripts/train_yolov8.py
from ultralytics import YOLO
import os

# ============================================================
# KONFIGURATION
# ============================================================
MODEL = "yolov8n.pt"   # oder yolov8m.pt für genauere Erkennung
DATA = "../dataset/data.yaml"
IMG_SIZE = 1024
EPOCHS = 4
BATCH = 8
DEVICE = "cpu"
TRAIN_NAME = "train_4_epoch_with_parameter"   # beliebiger Name für das Training

# Ordnerstruktur
PROJECT_DIR = "runs/detect/training"

# Falls der Ordner noch nicht existiert, wird er erstellt
os.makedirs(PROJECT_DIR, exist_ok=True)

# ============================================================
# TRAININGS-PARAMETER
# ============================================================
train_params = {
    # Grundparameter
    "task": "detect",
    "mode": "train",
    "data": DATA,
    "epochs": EPOCHS,
    "batch": BATCH,
    "imgsz": IMG_SIZE,
    "device": DEVICE,
    "name": TRAIN_NAME,
    "project": PROJECT_DIR,
    "save": True,
    "save_period": -1,
    "cache": True,
    "workers": 8,
    "pretrained": True,
    "seed": 0,
    "deterministic": True,
    "exist_ok": False,

    # Optimizer & Learning Rate
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "nbs": 64,
    "cos_lr": True,

    # Loss-Gewichte
    "box": 8.5,
    "cls": 1.0,
    "dfl": 2.0,

    # Augmentation
    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.6,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.001,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.1,
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "close_mosaic": 20,

    # Validation / Inference
    "val": True,
    "split": "val",
    "iou": 0.6,
    "max_det": 300,
    "patience": 40,
}

# ============================================================
# TRAINING
# ============================================================
model = YOLO(MODEL)
model.train(**train_params)

print(f"✅ Training abgeschlossen. Ergebnisse in: {os.path.join(PROJECT_DIR, TRAIN_NAME)}")