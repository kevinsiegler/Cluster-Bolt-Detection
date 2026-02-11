import os
import numpy as np
from tqdm import tqdm
from feature_extractor import read_points, extract_features
from config import TRAIN_LABELS, FEATURE_PATH

all_features = []

for file in tqdm(os.listdir(TRAIN_LABELS)):
    if not file.endswith(".txt"):
        continue

    pts, confs = read_points(os.path.join(TRAIN_LABELS, file))
    feats = extract_features(pts, confs)

    if feats is not None:
        all_features.append(feats)

all_features = np.vstack(all_features)
np.save(FEATURE_PATH, all_features)

print("Training features saved:", all_features.shape)
