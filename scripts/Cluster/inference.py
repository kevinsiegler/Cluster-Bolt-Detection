r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\inference.py
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist 
from utils import load_config, ensure_dir, align_points, save_yolo_labels, load_yolo_labels, find_best_match

def run_inference():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    run_name = cfg['inference'].get('run_name', 'default_run')
    
    # Paths
    model_name = cfg['clustering'].get('model_name', 'prototypes')
    model_path = os.path.join(cfg['paths']['output_root'], cfg['paths']['model_dir'], f"{model_name}.pkl")
    input_dir = cfg['paths']['inference_input_dir']
    
    output_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    
    ensure_dir(output_dir)
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        prototypes = pickle.load(f)
        
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    inlier_threshold = cfg['inference'].get('inlier_threshold', 0.022)
    acceptance_threshold = cfg['inference'].get('acceptance_threshold', 1.5)
    max_translation = cfg['inference'].get('max_translation', 0.03)
    filter_input = cfg['inference'].get('filter_input_points', False)
    def_w, def_h = cfg['inference']['default_box_size']
    
    print(f"Running inference on {len(files)} files...")
    
    for filename in tqdm(files):
        # Load Input (Observed Bolts)
        # This now loads from YOLO txt files, ignoring class.
        all_labels = load_yolo_labels(os.path.join(input_dir, filename))

        # Separate classes
        pts_0 = np.empty((0, 5)) # Class 0: Existing
        pts_1 = np.empty((0, 5)) # Class 1: YOLO Predicted Missing
        
        if len(all_labels) > 0:
            pts_0 = all_labels[all_labels[:, 0] == 0]
            pts_1 = all_labels[all_labels[:, 0] == 1]

        # Prepare matching points and weights
        # Weight 1.0 for Existing, 0.5 for Missing (heuristic for "higher weighting")
        match_pts_list = []
        match_weights_list = []
        
        if len(pts_0) > 0:
            match_pts_list.append(pts_0[:, 1:3]) # x,y
            match_weights_list.append(np.ones(len(pts_0)) * 1.0)
            
        if len(pts_1) > 0:
            match_pts_list.append(pts_1[:, 1:3]) # x,y
            match_weights_list.append(np.ones(len(pts_1)) * 0.5) # 0.5 outlier penalty for missing bolts
            
        if match_pts_list:
            match_pts = np.vstack(match_pts_list)
            match_weights = np.hstack(match_weights_list)
        else:
            match_pts = np.empty((0, 2))
            match_weights = np.array([])

        # Check if enough points for alignment
        # We use the combined count of existing + missing bolts.
        if len(match_pts) <= 1:
            # Fallback: If 0 or 1 bolt, keep YOLO result as is (don't try to cluster)
            save_yolo_labels(os.path.join(output_dir, filename), all_labels)
            continue
            
        # Use match_pts for alignment
        input_pts = match_pts 
            
        # 1. Find Best Prototype using shared logic
        missing_penalty = cfg['inference'].get('missing_penalty', 0.01)
        outlier_penalty = cfg['inference'].get('outlier_penalty', 1.0)
        best_proto, best_score = find_best_match(
            input_pts, prototypes, inlier_threshold, outlier_penalty=outlier_penalty, missing_penalty=missing_penalty, input_weights=match_weights
        )

        # --- ACCEPTANCE SCORE CALCULATION (NO ALIGNMENT) ---
        # After selecting the best prototype based on strict matching, we calculate a
        # score to decide if the match is good enough to be accepted.
        # This uses the untransformed prototype, respecting the 1:1 matching rule.
        best_aligned_proto = None
        final_score_for_acceptance = float('inf')

        if best_proto is not None:
            # The "aligned" prototype is simply the original, untransformed prototype.
            best_aligned_proto = best_proto['points'][:, :2]
            # The score calculated during selection IS the acceptance score
            final_score_for_acceptance = best_score
        
        # 2. Identify Missing Bolts
        predicted_missing = []
        
        # The main cause of high FP is accepting a bad prototype match.
        # The `best_score` is the average distance of input points to the aligned prototype.
        # A good match should have an average error significantly smaller than the individual point match_threshold.
        # We use `acceptance_threshold` to gate this.
        # Ensure threshold is large enough to cover "1% IoU" cases (approx 0.05 normalized)
        missing_detection_thresh = inlier_threshold*3

        input_rows = []
        # FIX: Use the new acceptance_threshold to decide if the match is good enough
        if best_proto is not None and final_score_for_acceptance < acceptance_threshold:
            # 1-to-1 Matching Logic (Greedy)
            # We only match against EXISTING bolts (pts_0) to decide what to keep.
            # Missing bolts (pts_1) were only for alignment and are now ignored.
            input_pts_0 = pts_0[:, 1:3] if len(pts_0) > 0 else np.empty((0, 2))
            dists = cdist(best_aligned_proto, input_pts_0) if len(input_pts_0) > 0 else np.full((len(best_aligned_proto), 0), float('inf'))
            
            dists_copy = dists.copy()
            matched_proto_indices = set()
            matched_input_indices = set()
            
            while True:
                # Find minimum distance in the matrix
                if dists_copy.size == 0 or np.all(np.isinf(dists_copy)):
                    break
                
                min_idx = np.unravel_index(np.argmin(dists_copy), dists_copy.shape)
                min_dist = dists_copy[min_idx]
                
                if min_dist > missing_detection_thresh:
                    break
                
                p_idx, i_idx = min_idx
                matched_proto_indices.add(p_idx)
                matched_input_indices.add(i_idx)
                
                # Mask out this prototype point and this input point
                dists_copy[p_idx, :] = float('inf')
                dists_copy[:, i_idx] = float('inf')
            
            # Any prototype point not matched is considered missing
            missing_indices = [i for i in range(len(best_aligned_proto)) if i not in matched_proto_indices]
            
            for idx in missing_indices:
                pt = best_aligned_proto[idx]

                # Use the average size of observed bolts for the predicted ones
                # This is much better than a fixed default size.
                # Prefer size from existing bolts, fallback to missing bolts, then default
                if len(pts_0) > 0:
                    avg_w = np.mean(pts_0[:, 3])
                    avg_h = np.mean(pts_0[:, 4])
                elif len(pts_1) > 0:
                    avg_w = np.mean(pts_1[:, 3])
                    avg_h = np.mean(pts_1[:, 4])
                else:
                    avg_w, avg_h = def_w, def_h

                # Check bounds
                if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                    predicted_missing.append([1, pt[0], pt[1], avg_w, avg_h])
            
            # Decide which input points to keep
            if filter_input:
                num_input_bolts = len(pts_0)
                num_to_remove = num_input_bolts - len(matched_input_indices)

                # NEU: Sicherheitsprüfung gegen exzessives Entfernen.
                # Wenn die Hälfte oder mehr der Schrauben entfernt werden sollen, ist das Match wahrscheinlich falsch.
                # In diesem Fall werden keine Schrauben entfernt, um False Positives zu vermeiden.
                if num_input_bolts > 0 and num_to_remove >= (num_input_bolts / 2):
                    # Behalte alle ursprünglichen Schrauben
                    input_rows = pts_0
                else:
                    # Entferne die nicht gematchten Schrauben wie geplant
                    kept_indices = sorted(list(matched_input_indices))
                    input_rows = pts_0[kept_indices] if len(kept_indices) > 0 else np.empty((0, 5))
            else:
                input_rows = pts_0
        
        else: # No good prototype match found
            if not filter_input:
                input_rows = pts_0
        
        predicted_missing = np.array(predicted_missing)

        # 3. Save Result
        if len(predicted_missing) > 0:
            final_output = np.vstack([input_rows, predicted_missing])
        else:
            final_output = np.array(input_rows)
            
        # Save YOLO txt
        save_yolo_labels(os.path.join(output_dir, filename), final_output)

    print(f"Inference complete. Results in {output_dir}")

if __name__ == "__main__":
    run_inference()
