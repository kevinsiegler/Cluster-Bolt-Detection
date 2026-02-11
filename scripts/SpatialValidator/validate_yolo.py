import os
import joblib
import numpy as np
from feature_extractor import extract_features
from config import YOLO_PRED_DEFAULT, MODEL_PATH
from datetime import datetime
import cv2

# -------------------------
# Benutzerdefinierter Pfad zu den Val-Bildern
IMG_DIR = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\images\val"
# -------------------------

# Zeitstempel für individuellen Lauf
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT = os.path.join(r"outputs", f"validation_{timestamp}")
LABEL_OUT = os.path.join(BASE_OUTPUT, "validated_labels")
IMAGE_OUT = os.path.join(BASE_OUTPUT, "validated_images")
os.makedirs(LABEL_OUT, exist_ok=True)
os.makedirs(IMAGE_OUT, exist_ok=True)

# Modell laden
model = joblib.load(MODEL_PATH)

# Alle YOLO-Vorhersagen durchlaufen
for file in os.listdir(YOLO_PRED_DEFAULT):
    if not file.endswith(".txt"):
        continue

    # Original YOLO Boxen laden
    original_boxes = []
    original_class_ids = []
    confs = []

    with open(os.path.join(YOLO_PRED_DEFAULT, file), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) == 6 else 1.0
            original_boxes.append([x_c, y_c, w, h])
            original_class_ids.append(cls)
            confs.append(conf)

    if len(original_boxes) == 0:
        continue  # kein Boxen in diesem Bild

    original_boxes = np.array(original_boxes, dtype=np.float32)
    confs = np.array(confs, dtype=np.float32)

    if original_boxes.ndim == 1:
        original_boxes = original_boxes.reshape(1, -1)
        confs = confs.reshape(1,)

    # Feature Extraktion (nur x_center, y_center + conf)
    feats = extract_features(original_boxes[:, :2], confs)
    if feats is None:
        continue

    # Anomaly Prediction
    preds = model.predict(feats)  # 1 = normal, -1 = anomal
    valid_indices = np.where(preds == 1)[0]

    # -------------------------
    # Validierte YOLO-kompatible Labels schreiben
    with open(os.path.join(LABEL_OUT, file), "w") as f:
        for idx in valid_indices:
            cls = original_class_ids[idx]
            x_c, y_c, w, h = original_boxes[idx]
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # -------------------------
    # Validierte Boxen auf Bild visualisieren
    img_name = file.replace(".txt", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"Bild nicht gefunden: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"cv2 konnte Bild nicht laden: {img_path}")
        continue

    h_img, w_img, _ = img.shape

    # Original YOLO Boxen grün
    for box in original_boxes:
        x1 = int((box[0]-box[2]/2)*w_img)
        y1 = int((box[1]-box[3]/2)*h_img)
        x2 = int((box[0]+box[2]/2)*w_img)
        y2 = int((box[1]+box[3]/2)*h_img)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)

    # Validierte Boxen blau
    for idx in valid_indices:
        box = original_boxes[idx]
        x1 = int((box[0]-box[2]/2)*w_img)
        y1 = int((box[1]-box[3]/2)*h_img)
        x2 = int((box[0]+box[2]/2)*w_img)
        y2 = int((box[1]+box[3]/2)*h_img)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

    # Entfernte Boxen rot
    removed_indices = [i for i in range(len(original_boxes)) if i not in valid_indices]
    for idx in removed_indices:
        box = original_boxes[idx]
        x1 = int((box[0]-box[2]/2)*w_img)
        y1 = int((box[1]-box[3]/2)*h_img)
        x2 = int((box[0]+box[2]/2)*w_img)
        y2 = int((box[1]+box[3]/2)*h_img)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

    # Bild speichern
    cv2.imwrite(os.path.join(IMAGE_OUT, img_name), img)

print(f"Spatial Validation abgeschlossen. Ergebnisse in {BASE_OUTPUT}")
