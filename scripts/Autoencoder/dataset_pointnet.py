import os
import torch
import numpy as np
from config import TRAIN_LABELS, VAL_LABELS, MAX_POINTS


def read_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float32)


class PointDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f)
                      for f in os.listdir(folder) if f.endswith('.txt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts = read_points(self.files[idx])

        # Sicherstellen, dass leere Dateien korrekt behandelt werden
        if pts.size == 0:
            pts = np.zeros((0, 2), dtype=np.float32)

        # Falls nur eine Dimension vorhanden ist (z.B. ein einzelner Punkt falsch gelesen)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)

        n = pts.shape[0]

        if n > MAX_POINTS:
            pts = pts[:MAX_POINTS]
        else:
            pad = np.zeros((MAX_POINTS - n, 2), dtype=np.float32)
            pts = np.vstack([pts, pad])

        return torch.from_numpy(pts), n


def get_datasets():
    train = PointDataset(TRAIN_LABELS)
    val = PointDataset(VAL_LABELS) if os.path.exists(VAL_LABELS) else None
    return train, val