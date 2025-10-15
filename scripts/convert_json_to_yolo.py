# scripts/convert_json_to_yolo.py
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# --------- CONFIG ---------
JSON_PATH = "../annotations.json"   # passe an
IMAGES_DIR = "../images"            # wo die .jpg/.png liegen
OUT_DIR = "../dataset"              # Ausgabeordner
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Mapping: original category_id -> YOLO class index
KEEP_CATEGORY_IDS = {1: 0, 2: 1}  # 1=Bolt->0, 2=Missing_Bolt->1

# --------------------------
random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)
for d in ["images/train","images/val","labels/train","labels/val"]:
    os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Expecting either a list of items or a dict with "images" / "annotations".
# Based on your snippet looks like top-level may be a list of objects per image.
# If your JSON structure differs, adapt this quickly.

# If file is COCO style with 'annotations' key at top-level:
if isinstance(data, dict) and "annotations" in data and "images" in data:
    annotations = data["annotations"]
    images_meta = {im["id"]: im for im in data["images"]}
else:
    # assume list of image objects like in your snippet
    annotations = []
    images_meta = {}
    for item in data:
        img_id = item["image_id"]
        images_meta[img_id] = {"file_name": f"{img_id}.jpg", "id": img_id}
        for ann in item.get("annotations", []):
            ann_copy = dict(ann)
            ann_copy["image_id"] = img_id
            annotations.append(ann_copy)

# group annotations by image_id
anns_by_image = {}
for ann in annotations:
    img_id = ann["image_id"]
    anns_by_image.setdefault(img_id, []).append(ann)

all_image_ids = list(anns_by_image.keys())
random.shuffle(all_image_ids)
split_idx = int(len(all_image_ids) * TRAIN_RATIO)
train_ids = set(all_image_ids[:split_idx])
val_ids   = set(all_image_ids[split_idx:])

def convert_bbox_xywh_to_yolo(x_min, y_min, w, h):
    x_center = x_min + w/2.0
    y_center = y_min + h/2.0
    return x_center, y_center, w, h

# process images
for img_id, anns in tqdm(anns_by_image.items(), desc="Images"):
    img_filename = images_meta.get(img_id, {}).get("file_name", f"{img_id}.jpg")
    src_img_path = os.path.join(IMAGES_DIR, img_filename)
    if not os.path.exists(src_img_path):
        # try jpg/png fallback
        if os.path.exists(src_img_path[:-4] + ".png"):
            src_img_path = src_img_path[:-4] + ".png"
        else:
            print(f"Warning: image not found {src_img_path}, skipping.")
            continue

    subset = "train" if img_id in train_ids else "val"
    dst_img_path = os.path.join(OUT_DIR, "images", subset, img_filename)
    shutil.copyfile(src_img_path, dst_img_path)

    label_lines = []
    for ann in anns:
        cid = ann.get("category_id")
        if cid not in KEEP_CATEGORY_IDS:
            continue
        yolo_cls = KEEP_CATEGORY_IDS[cid]
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        x_min, y_min, w, h = bbox  # we assume normalized x_min,y_min,w,h
        # safety: clamp to 0..1
        x_min = max(0.0, min(1.0, float(x_min)))
        y_min = max(0.0, min(1.0, float(y_min)))
        w = max(0.0, min(1.0, float(w)))
        h = max(0.0, min(1.0, float(h)))
        x_c, y_c, w, h = convert_bbox_xywh_to_yolo(x_min, y_min, w, h)
        # further clamp centers
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))
        label_lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    label_path = os.path.join(OUT_DIR, "labels", subset, img_filename.rsplit(".",1)[0] + ".txt")
    with open(label_path, "w", encoding="utf-8") as lf:
        lf.write("\n".join(label_lines))

print("Fertig. Dataset liegt in:", OUT_DIR)
