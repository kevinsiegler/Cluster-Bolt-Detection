import os

BASE_DIR = r"C:\Users\Kevin\Clustererkennung\bolt_detection"

TRAIN_LABELS = os.path.join(BASE_DIR, "dataset", "labels", "train")
VAL_LABELS = os.path.join(BASE_DIR, "dataset", "labels", "val")

YOLO_PRED_DEFAULT = r"C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\YOLO\testing\test_data-"

OUT_BASE = os.path.join(BASE_DIR, "scripts", "Autoencoder", "outputs")
MODEL_DIR = os.path.join(OUT_BASE, "model")
THRESH_DIR = os.path.join(OUT_BASE, "thresholds")
LABEL_OUT = os.path.join(OUT_BASE, "validated_labels")
IMAGE_OUT = os.path.join(OUT_BASE, "validated_images")

for p in [MODEL_DIR, THRESH_DIR, LABEL_OUT, IMAGE_OUT]:
    os.makedirs(p, exist_ok=True)

DEVICE = "cuda" if False else "cpu"
EPOCHS = 120
LR = 1e-3
BATCH_SIZE = 32
LATENT_DIM = 128
MAX_POINTS = 150  # obere Grenze Schraubenanzahl
MODEL_PATH = os.path.join(MODEL_DIR, "pointnet_best.pth")