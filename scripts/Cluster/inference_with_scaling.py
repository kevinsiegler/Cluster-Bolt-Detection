r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\inference_with_scaling.py
"""
import os
import numpy as np
import pickle
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist 
from utils import load_config, ensure_dir, save_yolo_labels, load_yolo_labels

def find_best_match_with_scaling(input_pts, prototypes, inlier_threshold, outlier_penalty=1.0, missing_penalty=0.01, input_weights=None):
    """
    GPU-beschleunigte Version des Exhaustive Matching mittels PyTorch.
    Probiert jeden Cluster-Punkt auf jeden Input-Punkt zu legen und testet verschiedene Skalierungen parallel.
    """
    # Prüfen ob GPU verfügbar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Daten auf Device schieben
    t_input_pts = torch.tensor(input_pts, dtype=torch.float32, device=device) # (N, 2)
    
    if input_weights is None:
        t_weights = torch.ones(len(input_pts), dtype=torch.float32, device=device)
    else:
        t_weights = torch.tensor(input_weights, dtype=torch.float32, device=device)
        
    scales = torch.arange(0.85, 1.16, 0.05, device=device) # (S,)
    
    best_score = float('inf')
    best_proto = None
    best_aligned_proto = None
    
    # Loop über Prototypen (diese sind unterschiedlich groß, daher schwer komplett zu batchen ohne Padding)
    for proto in prototypes:
        t_proto_pts = torch.tensor(proto['points'][:, :2], dtype=torch.float32, device=device) # (M, 2)
        
        M = t_proto_pts.shape[0]
        N = t_input_pts.shape[0]
        S = scales.shape[0]
        
        # --- Vektorisierung aller Kombinationen (M x N x S) ---
        
        # 1. Dimensionen vorbereiten für Broadcasting
        # Proto-Punkte (M, 2) -> (1, 1, 1, M, 2)
        proto_expanded = t_proto_pts.view(1, 1, 1, M, 2)
        # Ankerpunkte des Prototyps (M, 2) -> (M, 1, 1, 1, 2)
        p_anchors = t_proto_pts.view(M, 1, 1, 1, 2)
        # Ankerpunkte des Inputs (N, 2) -> (1, N, 1, 1, 2)
        i_anchors = t_input_pts.view(1, N, 1, 1, 2)
        # Skalierungen (S,) -> (1, 1, S, 1, 1)
        s_expanded = scales.view(1, 1, S, 1, 1)
        
        # Transformation: (proto - p_anchor) * scale + i_anchor
        # Shape: (M, N, S, M, 2)
        aligned_protos = (proto_expanded - p_anchors) * s_expanded + i_anchors
        
        # Flatten der Batch-Dimensionen: K = M * N * S
        K = M * N * S
        aligned_protos_flat = aligned_protos.view(K, M, 2)
        
        # 2. Distanzberechnung (Input vs. alle Kandidaten)
        # Input (N, 2) -> (1, N, 2) -> expand zu (K, N, 2)
        input_expanded = t_input_pts.view(1, N, 2).expand(K, N, 2)
        
        # cdist: (K, N, 2) vs (K, M, 2) -> (K, N, M)
        dists = torch.cdist(input_expanded, aligned_protos_flat)
        
        # 3. Scoring (Parallel für alle K Kandidaten)
        # min über dim 2 (M) -> (K, N)
        min_dists, closest_proto_indices = torch.min(dists, dim=2)
        
        # Inlier Maske
        inlier_mask = min_dists < inlier_threshold # (K, N)
        
        # Outlier Score
        weights_expanded = t_weights.view(1, N).expand(K, N)
        outlier_score = (weights_expanded * (~inlier_mask).float()).sum(dim=1) * outlier_penalty # (K,)
        
        # Inlier Distanz Mean
        inlier_dists_sum = (min_dists * inlier_mask.float()).sum(dim=1)
        num_inliers = inlier_mask.sum(dim=1)
        
        inlier_dist_mean = torch.zeros_like(outlier_score)
        mask_has_inliers = num_inliers > 0
        inlier_dist_mean[mask_has_inliers] = inlier_dists_sum[mask_has_inliers] / num_inliers[mask_has_inliers].float()
        
        # Missing Penalty (Num Predicted Missing)
        # One-Hot Encoding der Indices: (K, N, M)
        one_hot = F.one_hot(closest_proto_indices, num_classes=M).float()
        # Nur Inliers zählen
        one_hot_masked = one_hot * inlier_mask.unsqueeze(-1).float()
        # Summe über Input-Punkte -> Wie oft wurde jeder Proto-Punkt getroffen? (K, M)
        points_hit_counts = one_hot_masked.sum(dim=1)
        # Zählen wie viele > 0 sind (Unique Matches)
        num_unique_matched = (points_hit_counts > 0).sum(dim=1).float()
        
        num_predicted_missing = M - num_unique_matched
        
        scores = outlier_score + inlier_dist_mean + (num_predicted_missing * missing_penalty)
        
        # Besten Score in diesem Batch finden
        min_score_batch, min_idx_batch = torch.min(scores, dim=0)
        
        if min_score_batch.item() < best_score:
            best_score = min_score_batch.item()
            best_proto = proto
            # Zurück aufs CPU/Numpy Format für den Return
            best_aligned_proto = aligned_protos_flat[min_idx_batch].cpu().numpy()
            
    return best_proto, best_score, best_aligned_proto

def run_inference():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    
    # Append _scaled to run_name to differentiate from fixed inference
    run_name = cfg['inference'].get('run_name', 'default_run')
    
    # Paths
    model_name = cfg['clustering'].get('model_name', 'prototypes')
    model_path = os.path.join(cfg['paths']['output_root'], cfg['paths']['model_dir'], f"{model_name}.pkl")
    input_dir = cfg['paths']['inference_input_dir']
    
    output_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        prototypes = pickle.load(f)
        
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # Parameters
    # --- HARDCODED PARAMETERS FOR SCALING APPROACH ---
    # Wir entkoppeln dies von der Config, um für den Scaling-Ansatz robustere Werte zu nutzen.
    inlier_threshold = 0.05        # Sehr toleranter Bereich für das Finden des Clusters (Matching)
    acceptance_threshold = 9999.0  # Wir wollen fast IMMER das beste Cluster nehmen
    missing_penalty = 0.4          # Geringere Strafe, damit Cluster auch bei Lücken gewählt werden
    outlier_penalty = 0.6          # Geringere Strafe für Noise
    filter_input = False           # Keine YOLO-Schrauben entfernen (wie angefordert)
    # -------------------------------------------------

    def_w, def_h = cfg['inference']['default_box_size']
    
    print(f"Running inference with SCALING on {len(files)} files...")
    print(f"Using device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Output directory: {output_dir}")
    
    for filename in tqdm(files):
        # Load Input
        all_labels = load_yolo_labels(os.path.join(input_dir, filename))

        pts_0 = np.empty((0, 5)) 
        pts_1 = np.empty((0, 5)) 
        
        if len(all_labels) > 0:
            pts_0 = all_labels[all_labels[:, 0] == 0]
            pts_1 = all_labels[all_labels[:, 0] == 1]

        match_pts_list = []
        match_weights_list = []
        
        if len(pts_0) > 0:
            match_pts_list.append(pts_0[:, 1:3]) 
            match_weights_list.append(np.ones(len(pts_0)) * 1.0)
            
        if len(pts_1) > 0:
            match_pts_list.append(pts_1[:, 1:3]) 
            match_weights_list.append(np.ones(len(pts_1)) * 0.5) 
            
        if match_pts_list:
            match_pts = np.vstack(match_pts_list)
            match_weights = np.hstack(match_weights_list)
        else:
            match_pts = np.empty((0, 2))
            match_weights = np.array([])

        if len(match_pts) <= 1:
            save_yolo_labels(os.path.join(output_dir, filename), all_labels)
            continue
            
        input_pts = match_pts 
            
        # 1. Find Best Prototype with SCALING
        best_proto, best_score, best_aligned_proto = find_best_match_with_scaling(
            input_pts, prototypes, inlier_threshold, outlier_penalty=outlier_penalty, missing_penalty=missing_penalty, input_weights=match_weights
        )

        # 2. Identify Missing Bolts
        predicted_missing = []
        missing_detection_thresh = 0.05 # Threshold: Wenn Cluster-Punkt > 0.05 von existierender Schraube entfernt, dann hinzufügen.
        input_rows = []
        
        # Use the ALIGNED prototype for checking
        if best_proto is not None and best_score < acceptance_threshold:
            # 1-to-1 Matching Logic
            input_pts_0 = pts_0[:, 1:3] if len(pts_0) > 0 else np.empty((0, 2))
            
            # Prüfe jeden Punkt des Clusters: Ist eine YOLO-Schraube in der Nähe?
            dists = cdist(best_aligned_proto, input_pts_0) if len(input_pts_0) > 0 else np.full((len(best_aligned_proto), 0), float('inf'))
            
            # Für jeden Cluster-Punkt den Abstand zur nächsten existierenden Schraube finden
            if dists.shape[1] > 0:
                min_dists_to_existing = np.min(dists, axis=1)
            else:
                min_dists_to_existing = np.full(len(best_aligned_proto), float('inf'))

            for i, dist in enumerate(min_dists_to_existing):
                # Wenn der Abstand größer als der Threshold ist, fehlt die Schraube laut Cluster
                if dist > missing_detection_thresh:
                    pt = best_aligned_proto[i]
                    
                    # Größe schätzen
                    if len(pts_0) > 0:
                        avg_w, avg_h = np.mean(pts_0[:, 3]), np.mean(pts_0[:, 4])
                    elif len(pts_1) > 0:
                        avg_w, avg_h = np.mean(pts_1[:, 3]), np.mean(pts_1[:, 4])
                    else:
                        avg_w, avg_h = def_w, def_h

                    if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                        predicted_missing.append([1, pt[0], pt[1], avg_w, avg_h])
            
            # Wir behalten IMMER alle existierenden YOLO-Schrauben (filter_input = False)
            input_rows = pts_0
        else: 
            # Fallback: Wenn kein Cluster passt, behalten wir nur die YOLO-Schrauben
            input_rows = pts_0
        
        # Zusammenfügen: Existierende (YOLO) + Neu berechnete Fehlende (Cluster)
        # Alte YOLO-Fehlende (pts_1) werden ignoriert/überschrieben
        predicted_missing = np.array(predicted_missing)
        if len(predicted_missing) > 0:
            final_output = np.vstack([input_rows, predicted_missing])
        else:
            final_output = np.array(input_rows)
            
        save_yolo_labels(os.path.join(output_dir, filename), final_output)

    print(f"Inference complete. Results in {output_dir}")

if __name__ == "__main__":
    run_inference()