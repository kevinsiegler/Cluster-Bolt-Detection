import os
import cv2
import numpy as np
from config import YOLO_PRED_DEFAULT, LABEL_OUT, IMAGE_OUT

IMG_DIR = r"../dataset/images/val"  # Pfad zu den Bildern

os.makedirs(IMAGE_OUT, exist_ok=True)

for file in os.listdir(YOLO_PRED_DEFAULT):
    if not file.endswith(".txt"):
        continue

    img_name = file.replace(".txt", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # YOLO Boxen laden
    yolo_boxes = []
    with open(os.path.join(YOLO_PRED_DEFAULT, file)) as f:
        for line in f:
            parts = line.strip().split()
            x_c, y_c, bw, bh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) == 6 else 1.0
            x1 = int((x_c - bw/2)*w)
            y1 = int((y_c - bh/2)*h)
            x2 = int((x_c + bw/2)*w)
            y2 = int((y_c + bh/2)*h)
            yolo_boxes.append((x1,y1,x2,y2))

    # Validierte Boxen laden
    valid_boxes = []
    with open(os.path.join(LABEL_OUT, file)) as f:
        for line in f:
            parts = line.strip().split()
            x_c, y_c, bw, bh = map(float, parts[1:5])
            x1 = int((x_c - bw/2)*w)
            y1 = int((y_c - bh/2)*h)
            x2 = int((x_c + bw/2)*w)
            y2 = int((y_c + bh/2)*h)
            valid_boxes.append((x1,y1,x2,y2))

    # Alle YOLO Boxen zeichnen (grün)
    for box in yolo_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)

    # Validierte Boxen zeichnen (blau)
    for box in valid_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)

    # Entfernte Boxen (rot)
    removed_boxes = [b for b in yolo_boxes if b not in valid_boxes]
    for box in removed_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)

    cv2.imwrite(os.path.join(IMAGE_OUT, img_name), img)

print("Visualisierung abgeschlossen. Ergebnisse in outputs/validated_images/")
