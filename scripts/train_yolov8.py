# scripts/train_yolov8.py
from ultralytics import YOLO

# Modellwahl: startr mit yolov8n (nano) oder yolov8m (medium) für bessere small-object Erkennung.
MODEL = "yolov8n.pt"  # falls RAM/GPU limitiert: "yolov8n.pt"
DATA = "../dataset/data.yaml"
IMG_SIZE = 1024    # größere Auflösung hilft bei sehr kleinen Objekten
EPOCHS = 30 
BATCH = 8          # an GPU memory anpassen, CPU: setze klein
DEVICE = "cpu"         # GPU id oder "cpu"

model = YOLO(MODEL)
model.train(data=DATA, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=DEVICE, name="bolt_run01")
