import os

BASE_DIR = r"C:\Users\Kevin\Clustererkennung\bolt_detection"

TRAIN_LABELS = os.path.join(BASE_DIR, "dataset", "labels", "train")
VAL_LABELS = os.path.join(BASE_DIR, "dataset", "labels", "val")

YOLO_PRED_DEFAULT = r"C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\YOLO\testing\test_data"

OUT_BASE = os.path.join(BASE_DIR, "scripts", "SpatialValidator", "outputs")
MODEL_DIR = os.path.join(OUT_BASE, "model")
FEATURE_DIR = os.path.join(OUT_BASE, "features")
LABEL_OUT = os.path.join(OUT_BASE, "validated_labels")
IMAGE_OUT = os.path.join(OUT_BASE, "validated_images")

for p in [MODEL_DIR, FEATURE_DIR, LABEL_OUT, IMAGE_OUT]:
    os.makedirs(p, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
FEATURE_PATH = os.path.join(FEATURE_DIR, "train_features.npy")

K_NEIGHBORS = 5
CONTAMINATION = 0.005  # erwartet extrem wenige Anomalien
