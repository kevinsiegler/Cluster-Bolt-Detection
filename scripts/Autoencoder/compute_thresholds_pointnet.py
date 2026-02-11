import torch
import json
from dataset_pointnet import get_datasets
from model_pointnet import PointNetAE, chamfer_distance
from config import *


def compute():
    _, val_ds = get_datasets()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=1)

    model = PointNetAE(LATENT_DIM, MAX_POINTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    scores = []
    with torch.no_grad():
        for pts, n in loader:
            pts = pts.to(DEVICE)
            recon = model(pts)
            score = chamfer_distance(pts, recon).item()
            scores.append(score)

    import numpy as np
    mean = np.mean(scores)
    std = np.std(scores)
    thresh = float(mean + 2.5 * std)

    with open(os.path.join(THRESH_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": thresh}, f, indent=2)

    print("Threshold:", thresh)


if __name__ == '__main__':
    compute()