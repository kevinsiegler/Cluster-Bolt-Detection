r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\inference_with_scaling.py
"""
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm
from scipy.spatial.distance import cdist 
from utils import load_config, ensure_dir, save_yolo_labels, load_yolo_labels

def find_best_match_with_scaling(input_pts, prototypes, inlier_threshold, outlier_penalty=1.0, missing_penalty=0.01, input_weights=None):
    """
    Exhaustive Matching: Probiert jeden Cluster-Punkt auf jeden Input-Punkt zu legen
    und testet verschiedene Skalierungen. Robust gegen Ghost-Boxen.
    """
    best_score = float('inf')
    best_proto = None
    best_aligned_proto = None

    if input_weights is None:
        input_weights = np.ones(len(input_pts))
    
    # Skalierungen, die getestet werden sollen (85% bis 115%)
    scales = np.arange(0.85, 1.16, 0.05)

    # Optimierung: Nur Prototypen prüfen, die grob passen könnten (optional, hier deaktiviert für maximale Genauigkeit)
    for proto in prototypes:
        proto_pts = proto['points'][:, :2]
        
        # BRUTE FORCE MATCHING:
        # Wir probieren JEDEN Cluster-Punkt auf JEDEN Input-Punkt zu legen.
        for p_idx, p_anchor in enumerate(proto_pts):
            # Iteriere über jeden Punkt im Input (Anker I)
            for i_idx, i_anchor in enumerate(input_pts):
                # Iteriere über Skalierungen
                for s in scales:
                    # Transformation:
                    # 1. Verschiebe Prototyp so, dass p_anchor im Ursprung ist (proto_pts - p_anchor)
                    # 2. Skaliere (* s)
                    # 3. Verschiebe zum Input-Anker (+ i_anchor)
                    aligned_proto = (proto_pts - p_anchor) * s + i_anchor
                    
                    # --- Score Berechnung ---
                    dists = cdist(input_pts, aligned_proto) # Distanzmatrix: Input x Cluster
                    
                    # Für jeden Input-Punkt den nächsten Cluster-Punkt finden
                    closest_proto_indices = np.argmin(dists, axis=1)
                    min_dists = dists[np.arange(len(input_pts)), closest_proto_indices]
                    
                    inlier_mask = min_dists < inlier_threshold
                    outlier_mask = ~inlier_mask
                    
                    outlier_score = np.sum(input_weights[outlier_mask]) * outlier_penalty
                    
                    num_inliers = np.sum(inlier_mask)
                    inlier_dist_mean = np.mean(min_dists[inlier_mask]) if num_inliers > 0 else 0
                    
                    # Wie viele Cluster-Punkte wurden getroffen?
                    # (Verhindert, dass ein Cluster-Punkt mehrere Input-Punkte "aufsaugt")
                    num_unique_matched = len(np.unique(closest_proto_indices[inlier_mask])) if num_inliers > 0 else 0
                    num_predicted_missing = len(proto_pts) - num_unique_matched
                    
                    score = outlier_score + inlier_dist_mean + (num_predicted_missing * missing_penalty)

                    if score < best_score:
                        best_score = score
                        best_proto = proto
                        best_aligned_proto = aligned_proto
            
    return best_proto, best_score, best_aligned_proto

def run_inference():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    
    # Append _scaled to run_name to differentiate from fixed inference
    run_name = cfg['inference'].get('run_name', 'default_run')
    run_name = cfg['inference'].get('run_name', 'default_run') + "_scaled"
    
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