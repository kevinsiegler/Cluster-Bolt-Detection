# scripts/infer.py
from ultralytics import YOLO
import cv2, os

MODEL_PATH = "runs/detect/bolt_run012/weights/best.pt"  # Pfad nach Training anpassen
IMG = "../dataset/images/val"   # oder einzelnes Bild

model = YOLO(MODEL_PATH)
results = model.predict(source=IMG, conf=0.25, imgsz=1024, save=True, save_txt=True) 
# save -> erzeugt Bilder mit Boxes in runs/detect/predict...
