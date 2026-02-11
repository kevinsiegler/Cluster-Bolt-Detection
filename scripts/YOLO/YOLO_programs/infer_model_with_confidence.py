from ultralytics import YOLO
import os

TRAIN_NAME = "train_30_epoch"
MODEL_PATH = f"runs/detect/training/{TRAIN_NAME}/weights/best.pt"
IMG = "../dataset/images/val"

# Um exakt die gleichen Labels wie infer_model.py zu erhalten, muss die Confidence übereinstimmen.
CONF = 0.4

OUTPUT_NAME = f"infer_{TRAIN_NAME}_conf({CONF})"
EVAL_BASE = "runs/detect/evaluations_w_confidence_txt_data"
os.makedirs(EVAL_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(EVAL_BASE, OUTPUT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMG,
    conf=CONF,
    imgsz=1024,
    save=False
)

for result in results:
    img_name = os.path.splitext(os.path.basename(result.path))[0]
    txt_path = os.path.join(OUTPUT_DIR, f"{img_name}.txt")

    with open(txt_path, "w") as f:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # YOLO-normalisierte Werte (0–1)
            x_center, y_center, w, h = box.xywhn[0].tolist()

            f.write(
                f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f}\n"
            )
