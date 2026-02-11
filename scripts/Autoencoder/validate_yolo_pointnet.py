import os
import torch
import json
import numpy as np
from PIL import Image, ImageDraw
from config import *
from model_pointnet import PointNetAE, chamfer_distance

# Set your image folder
IMAGE_FOLDER = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\images\val"


def read_points(path):
    pts = []
    raw = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                # Skip invalid lines
                continue
            raw.append(parts)
            pts.append([float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float32), raw


def validate_folder(yolo_folder, image_folder=None):
    # Load threshold
    thresh_path = os.path.join(THRESH_DIR, "threshold.json")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError(f"Threshold file not found: {thresh_path}")
    thresh = json.load(open(thresh_path))['threshold']

    # Load model
    model = PointNetAE(LATENT_DIM, MAX_POINTS).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Make sure output directories exist
    os.makedirs(LABEL_OUT, exist_ok=True)
    if image_folder:
        os.makedirs(IMAGE_OUT, exist_ok=True)

    # Process YOLO predictions
    for file in os.listdir(yolo_folder):
        if not file.endswith('.txt'):
            continue

        file_path = os.path.join(yolo_folder, file)
        pts, raw = read_points(file_path)

        if len(pts) == 0:
            print(f"Skipping empty or invalid file: {file}")
            continue  # skip empty files

        n = len(pts)
        if n > MAX_POINTS:
            print(f"Warning: {file} has {n} points, truncating to MAX_POINTS={MAX_POINTS}")
            pts = pts[:MAX_POINTS]
            n = MAX_POINTS

        pad = np.zeros((MAX_POINTS - n, 2), dtype=np.float32)
        pts_pad = np.vstack([pts, pad])

        tensor = torch.from_numpy(pts_pad).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = model(tensor)
            score = chamfer_distance(tensor, recon).item()

        keep = score <= thresh

        # Save labels if kept
        if keep:
            with open(os.path.join(LABEL_OUT, file), 'w') as f:
                for r in raw:
                    f.write(" ".join(r) + "\n")

        # Optional visualization
        if image_folder:
            img_name = file.replace('.txt', '.jpg')
            img_path = os.path.join(image_folder, img_name)
            if not os.path.exists(img_path):
                print(f"Image not found for {file}: {img_path}")
                continue

            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            w, h = img.size
            color = 'green' if keep else 'red'
            for r in raw:
                x = float(r[1])
                y = float(r[2])
                bw = float(r[3])
                bh = float(r[4])
                left = (x - bw/2) * w
                top = (y - bh/2) * h
                right = (x + bw/2) * w
                bottom = (y + bh/2) * h
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
            img.save(os.path.join(IMAGE_OUT, img_name))
            print(f"Saved visualization: {os.path.join(IMAGE_OUT, img_name)}")


if __name__ == '__main__':
    validate_folder(YOLO_PRED_DEFAULT, image_folder=IMAGE_FOLDER)
