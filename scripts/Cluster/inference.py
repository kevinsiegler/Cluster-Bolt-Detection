r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\inference.py
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist 
from utils import load_config, ensure_dir, align_points, save_yolo_labels, load_yolo_labels

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
    
    match_thresh = cfg['inference']['match_threshold']
    allow_scaling = cfg['inference']['allow_scaling']
    filter_input = cfg['inference'].get('filter_input_points', False)
    def_w, def_h = cfg['inference']['default_box_size']
    
    print(f"Running inference on {len(files)} files...")
    
    for filename in tqdm(files):
        # Load Input (Observed Bolts)
        # This now loads from YOLO txt files, ignoring class.
        all_labels = load_yolo_labels(os.path.join(input_dir, filename))

        # --- NEU: Filtere nur nach Klasse 0 (vorhandene Schrauben) als Input ---
        # Alle bereits existierenden Klasse-1-Labels aus der Eingabe werden ignoriert.
        if len(all_labels) > 0:
            labels = all_labels[all_labels[:, 0] == 0]
        else:
            labels = np.empty((0, 5))

        # NEU: Wenn 2 oder weniger Schrauben vorhanden sind, reicht es nicht für eine
        # zuverlässige Cluster-Prognose. In diesem Fall wird die Original-Datei
        # ohne Änderungen übernommen, da die Vorhersage fast immer falsch wäre.
        if len(labels) <= 2:
            save_yolo_labels(os.path.join(output_dir, filename), all_labels)
            continue
        input_data = labels[:, 1:5] # Use x,y,w,h

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
        
        # PROBLEM 1 FIX: Use a looser threshold for deciding if a specific point is missing.
        # Allow slight shifts (perspective/cutout) to still count as "present".
        missing_detection_thresh = match_thresh * 2

        input_rows = []
        if best_proto is not None and best_score < match_thresh:
            # 1-to-1 Matching Logic (Greedy)
            dists = cdist(best_aligned_proto, input_pts)
            dists_copy = dists.copy()
            matched_proto_indices = set()
            matched_input_indices = set()
            
            while True:
                # Find minimum distance in the matrix
                if np.all(np.isinf(dists_copy)):
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
                avg_w = np.mean(input_data[:, 2]) if input_data.shape[0] > 0 else def_w
                avg_h = np.mean(input_data[:, 3]) if input_data.shape[0] > 0 else def_h

                # Check bounds
                if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                    predicted_missing.append([1, pt[0], pt[1], avg_w, avg_h])
            
            # Decide which input points to keep
            if filter_input:
                num_input_bolts = input_data.shape[0]
                num_to_remove = num_input_bolts - len(matched_input_indices)

                # NEU: Sicherheitsprüfung gegen exzessives Entfernen.
                # Wenn die Hälfte oder mehr der Schrauben entfernt werden sollen, ist das Match wahrscheinlich falsch.
                # In diesem Fall werden keine Schrauben entfernt, um False Positives zu vermeiden.
                if num_input_bolts > 0 and num_to_remove >= (num_input_bolts / 2):
                    # Behalte alle ursprünglichen Schrauben
                    input_rows = [[0, row[0], row[1], row[2], row[3]] for row in input_data]
                else:
                    # Entferne die nicht gematchten Schrauben wie geplant
                    kept_indices = sorted(list(matched_input_indices))
                    input_rows = [[0, row[0], row[1], row[2], row[3]] for row in input_data[kept_indices]]
            else:
                input_rows = [[0, row[0], row[1], row[2], row[3]] for row in input_data]
        
        else: # No good prototype match found
            if not filter_input:
                input_rows = [[0, row[0], row[1], row[2], row[3]] for row in input_data]
        
        predicted_missing = np.array(predicted_missing)

        # --- NEU: Sicherheitsprüfung gegen exzessive Ergänzungen ---
        # Wenn mehr als 4x so viele Schrauben ergänzt wie vorhanden sind, ist es wahrscheinlich ein Fehlmatch.
        # In diesem Fall werden keine Schrauben ergänzt.
        num_input = input_data.shape[0]
        num_predicted = predicted_missing.shape[0]
        if num_input > 0 and num_predicted > (4 * num_input):
            # print(f"  -> Warnung: Exzessive Ergänzung ({num_predicted} > 4 * {num_input}). Verwerfe Ergänzungen für {filename}.")
            predicted_missing = np.array([])

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
