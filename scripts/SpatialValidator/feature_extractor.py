import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import K_NEIGHBORS


def read_points(path):
    pts = []
    confs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                pts.append([float(parts[1]), float(parts[2])])
                if len(parts) == 6:
                    confs.append(float(parts[5]))
                else:
                    confs.append(1.0)
    return np.array(pts), np.array(confs)


def extract_features(points, confs):
    if len(points) < 2:
        return None

    nbrs = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(points))).fit(points)
    distances, _ = nbrs.kneighbors(points)

    features = []
    for i in range(len(points)):
        d = distances[i]

        feat = [
            np.mean(d),
            np.std(d),
            np.min(d),
            np.max(d),
            len(points),
            confs[i]
        ]

        features.append(feat)

    return np.array(features)
