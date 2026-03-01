r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\utils.py
"""
import os
import yaml
import numpy as np
from scipy.spatial.distance import cdist

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_yolo_labels(path):
    """
    Loads YOLO labels. 
    Returns: numpy array [class, x, y, w, h]
    """
    if not os.path.exists(path):
        return np.empty((0, 5))
    
    labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])
    return np.array(labels)

def save_yolo_labels(path, labels):
    """
    Saves numpy array [class, x, y, w, h] to file.
    """
    with open(path, 'w') as f:
        for row in labels:
            f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")

def extract_point_features(points):
    """
    Extracts a feature vector from a point set for clustering.
    Features: [count, aspect_ratio, density, std_x, std_y]
    """
    if len(points) == 0:
        return np.zeros(5)
    
    count = len(points)
    
    # Bounding box
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    dims = max_xy - min_xy
    aspect_ratio = dims[0] / (dims[1] + 1e-6)
    
    # Spread
    std_xy = np.std(points, axis=0)
    
    # Density (points per area)
    area = (dims[0] * dims[1]) + 1e-6
    density = count / area
    
    return np.array([count, aspect_ratio, density, std_xy[0], std_xy[1]])

def align_points(source, target, allow_scaling=False):
    """
    Aligns 'source' points to 'target' points using centroid alignment.
    Returns: transformed_source, translation_vector
    """
    if len(source) == 0 or len(target) == 0:
        return source, np.zeros(2)
        
    centroid_src = np.mean(source, axis=0)
    centroid_tgt = np.mean(target, axis=0)
    
    translation = centroid_tgt - centroid_src
    
    aligned_src = source + translation
    
    if allow_scaling:
        # Simple scale estimation based on average radius from centroid
        rad_src = np.mean(np.linalg.norm(source - centroid_src, axis=1))
        rad_tgt = np.mean(np.linalg.norm(target - centroid_tgt, axis=1))
        
        if rad_src > 1e-4:
            scale = rad_tgt / rad_src
            # Re-calculate alignment with scale
            aligned_src = (source - centroid_src) * scale + centroid_tgt
            
    return aligned_src, translation

def chamfer_distance(set_a, set_b):
    """
    Computes bidirectional Chamfer distance between two point sets.
    Lower is more similar.
    """
    if len(set_a) == 0 or len(set_b) == 0:
        return float('inf')
        
    # Distance matrix
    dists = cdist(set_a, set_b, metric='euclidean')
    
    # Min dist from A to B
    min_a_to_b = np.mean(np.min(dists, axis=1))
    
    # Min dist from B to A
    min_b_to_a = np.mean(np.min(dists, axis=0))
    
    return min_a_to_b + min_b_to_a

def find_best_match(input_pts, prototypes, inlier_threshold, outlier_penalty=1.0, missing_penalty=0.01, input_weights=None):
    """
    Findet den besten passenden Prototypen für die gegebenen Input-Punkte.
    Zentralisierte Logik für Inferenz und Dashboard.
    
    Returns: (best_proto, best_score)
    """
    best_score = float('inf')
    best_proto = None

    if input_weights is None:
        input_weights = np.ones(len(input_pts))
    
    for proto in prototypes:
        proto_pts = proto['points'][:, :2]
        
        # Berechne Distanzen direkt (ohne Verschiebung/Alignment)
        dists = cdist(input_pts, proto_pts)
        
        # Finde den nächsten Prototyp-Punkt für jeden Input-Punkt
        closest_proto_indices = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(input_pts)), closest_proto_indices]
        
        # Bestimme Inliers basierend auf dem Threshold
        inlier_mask = min_dists < inlier_threshold
        outlier_mask = ~inlier_mask
        num_inliers = np.sum(inlier_mask)
        inlier_dist_mean = np.mean(min_dists[inlier_mask]) if num_inliers > 0 else 0
        
        # Weighted outlier score
        outlier_score = np.sum(input_weights[outlier_mask]) * outlier_penalty

        # Berechne Anzahl der vorhergesagten fehlenden Schrauben (Occam's Razor)
        num_unique_matched = len(np.unique(closest_proto_indices[inlier_mask])) if num_inliers > 0 else 0
        num_predicted_missing = len(proto_pts) - num_unique_matched
        
        score = outlier_score + inlier_dist_mean + (num_predicted_missing * missing_penalty)

        if score < best_score:
            best_score = score
            best_proto = proto
            
    return best_proto, best_score
