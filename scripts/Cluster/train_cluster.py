r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\train_cluster.py
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils import load_config, extract_point_features, ensure_dir, chamfer_distance

def train():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    
    data_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "train")
    model_dir = os.path.join(cfg['paths']['output_root'], cfg['paths']['model_dir'])
    ensure_dir(model_dir)
    
    print("--- Loading Training Data ---")
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    all_points = []
    features = []
    valid_files = []
    
    for f in tqdm(files):
        pts = np.load(os.path.join(data_dir, f))
        if len(pts) < 2: continue # Ignore noise/single points
        
        all_points.append(pts)
        # Features are based on x,y geometry only
        features.append(extract_point_features(pts[:, :2]))
        valid_files.append(f)
        
    features = np.array(features)
    print(f"Loaded {len(features)} valid layouts.")
    
    # 1. Clustering
    # We use K-Means on features to group similar layouts efficiently
    n_clusters = cfg['clustering']['n_clusters']
    if n_clusters > len(features):
        print(f"Warning: n_clusters ({n_clusters}) > n_samples ({len(features)}). Adjusting n_clusters to {len(features)}.")
        n_clusters = len(features)

    print(f"Clustering into {n_clusters} prototypes...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg['clustering']['random_state'], n_init=10)
    labels = kmeans.fit_predict(features)
    
    # 2. Prototype Selection (Medoids)
    # For each cluster, find the sample that is most representative (lowest avg Chamfer distance to others in cluster)
    # To save time, we just pick the one closest to the feature centroid, 
    # OR we compute pairwise chamfer for the top candidates.
    
    prototypes = []
    
    print("Selecting Prototypes...")
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) == 0: continue
        
        cluster_points = [all_points[idx] for idx in indices]
        
        # Simple Medoid: Find sample closest to feature center
        # (This is a heuristic approximation to avoid O(N^2) chamfer calc)
        center = kmeans.cluster_centers_[i]
        cluster_feats = features[indices]
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        medoid_idx = np.argmin(dists)
        
        best_layout = cluster_points[medoid_idx]
        prototypes.append({
            'id': i,
            'points': best_layout,
            'count': len(indices),
            'features': cluster_feats[medoid_idx]
        })
        
    print(f"Generated {len(prototypes)} initial prototypes.")
    
    # 3. Pruning (Optional but Recommended)
    # With many clusters, some might be very similar. Let's remove redundant ones.
    pruning_thresh = cfg['clustering'].get('pruning_threshold', 0.01)
    final_prototypes = []
    # Sort prototypes by how many samples they represent (most important first)
    prototypes.sort(key=lambda p: p['count'], reverse=True)

    print(f"Pruning prototypes with threshold {pruning_thresh}...")

    for proto in tqdm(prototypes):
        is_redundant = False
        # Compare to the already selected final prototypes
        for final_proto in final_prototypes:
            # Use only x,y for distance check
            dist = chamfer_distance(proto['points'][:, :2], final_proto['points'][:, :2])
            if dist < pruning_thresh:
                is_redundant = True
                break
        
        if not is_redundant:
            final_prototypes.append(proto)
    
    # Save Model
    model_name = cfg['clustering'].get('model_name', 'prototypes')
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(final_prototypes, f)
        
    print(f"Pruning complete. Saved {len(final_prototypes)} unique prototypes to {model_path}")

if __name__ == "__main__":
    train()
