r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\inference.py
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils import load_config, ensure_dir, align_points, save_yolo_labels

def run_inference():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    run_name = cfg['inference'].get('run_name', 'default_run')
    
    # Paths
    model_name = cfg['clustering'].get('model_name', 'prototypes')
    model_path = os.path.join(cfg['paths']['output_root'], cfg['paths']['model_dir'], f"{model_name}.pkl")
    input_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_input")
    
    # If input dir is empty/doesn't exist, maybe user wants to run on raw YOLO txts?
    # For this script, we assume preprocessing ran. 
    # To run on raw YOLO output, one would point input_dir there and add a loader check.
    
    output_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    
    ensure_dir(output_dir)
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        prototypes = pickle.load(f)
        
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    match_thresh = cfg['inference']['match_threshold']
    allow_scaling = cfg['inference']['allow_scaling']
    def_w, def_h = cfg['inference']['default_box_size']
    
    print(f"Running inference on {len(files)} files...")
    
    for filename in tqdm(files):
        # Load Input (Observed Bolts)
        # This now assumes the .npy file contains [x, y, w, h] from preprocessing
        input_data = np.load(os.path.join(input_dir, filename))
        
        if input_data.shape[0] < 2 or input_data.shape[1] < 2:
            # Not enough points to match structure
            # Save empty result or copy input
            output_rows = []
            if input_data.shape[0] > 0 and input_data.shape[1] >= 4:
                 output_rows = [[0, row[0], row[1], row[2], row[3]] for row in input_data]
            elif input_data.shape[0] > 0: # Fallback for old format
                 output_rows = [[0, row[0], row[1], def_w, def_h] for row in input_data]

            save_yolo_labels(os.path.join(output_dir, filename.replace('.npy', '.txt')), np.array(output_rows))
            continue

        # Use only x,y for geometric matching
        input_pts = input_data[:, :2]
            
        best_score = float('inf')
        best_proto = None
        best_translation = None
        best_aligned_proto = None
        
        # 1. Find Best Prototype
        # Iterate all prototypes, align, measure fit
        for proto in prototypes:
            # proto['points'] contains x,y,w,h. Use only x,y for geometric matching.
            proto_pts = proto['points'][:, :2]
            
            # Optimization: Skip if point counts are vastly different? 
            # No, because input is partial. Proto must have >= input points ideally.
            if len(proto_pts) < len(input_pts):
                continue
                
            # Align Prototype to Input
            # PROBLEM: Centroid alignment fails for partial inputs (centroid of subset != centroid of whole).
            # SOLUTION: Try to align by matching points directly (Brute-force translation).
            
            current_proto_best_score = float('inf')
            current_proto_best_aligned = None

            if allow_scaling:
                 # Fallback to centroid alignment if scaling is required (harder to brute force)
                 aligned_proto, trans = align_points(proto_pts, input_pts, allow_scaling=True)
                 dists = cdist(input_pts, aligned_proto)
                 score = np.mean(np.min(dists, axis=1))
                 current_proto_best_score = score
                 current_proto_best_aligned = aligned_proto
            else:
                # Try aligning every input point to every prototype point to find the best translation
                # This is O(N*M) but robust for partial data.
                for i in range(len(input_pts)):
                    for j in range(len(proto_pts)):
                        # Translation: Move Proto[j] to Input[i]
                        t = input_pts[i] - proto_pts[j]
                        candidate_proto = proto_pts + t
                        
                        # Score: How well does Input fit this shifted Prototype?
                        dists = cdist(input_pts, candidate_proto)
                        min_dists = np.min(dists, axis=1)
                        score = np.mean(min_dists)
                        
                        if score < current_proto_best_score:
                            current_proto_best_score = score
                            current_proto_best_aligned = candidate_proto

            if current_proto_best_score < best_score:
                best_score = current_proto_best_score
                best_proto = proto
                best_aligned_proto = current_proto_best_aligned
        
        # 2. Identify Missing Bolts
        predicted_missing = []
        
        # The main cause of high FP is accepting a bad prototype match.
        # The `best_score` is the average distance of input points to the aligned prototype.
        # A good match should have an average error significantly smaller than the individual point match_threshold.
        # We make the check much stricter to reject ambiguous or poor alignments.
        if best_proto is not None and best_score < match_thresh:
            # Check which points in the Best Prototype do NOT have a match in Input
            dists = cdist(best_aligned_proto, input_pts)
            # For each proto point, how far is the nearest input point?
            min_dists_proto_to_input = np.min(dists, axis=1)
            
            # If distance is large, it's missing in the input
            missing_indices = np.where(min_dists_proto_to_input > match_thresh)[0]
            
            for idx in missing_indices:
                pt = best_aligned_proto[idx]

                # Use the average size of observed bolts for the predicted ones
                # This is much better than a fixed default size.
                avg_w = np.mean(input_data[:, 2]) if input_data.shape[0] > 0 else def_w
                avg_h = np.mean(input_data[:, 3]) if input_data.shape[0] > 0 else def_h

                # Check bounds
                if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                    predicted_missing.append([1, pt[0], pt[1], avg_w, avg_h])
        
        predicted_missing = np.array(predicted_missing)
        
        # 3. Save Result
        # Reconstruct input rows (Class 0) using their ORIGINAL w,h
        input_rows = [[0, row[0], row[1], row[2], row[3]] for row in input_data]
        
        if len(predicted_missing) > 0:
            final_output = np.vstack([input_rows, predicted_missing])
        else:
            final_output = np.array(input_rows)
            
        # Save YOLO txt
        save_yolo_labels(os.path.join(output_dir, filename.replace('.npy', '.txt')), final_output)

    print(f"Inference complete. Results in {output_dir}")

if __name__ == "__main__":
    run_inference()
