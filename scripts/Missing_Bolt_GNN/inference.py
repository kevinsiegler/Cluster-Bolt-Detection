r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\inference.py
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from utils import CONFIG, load_yolo_labels, save_yolo_labels, setup_run_directories
from graph_builder import build_graph_from_points
from model import BoltCompletionGNN

def generate_grid_candidates(points, density=20, padding=0.15):
    """
    Generates a grid of points within the bounding box of the given points, with padding.
    """
    if len(points) < 2:
        # Fallback to a full grid if there's not enough points to define a region
        x = np.linspace(0, 1, density)
        y = np.linspace(0, 1, density)
    else:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        
        # Add padding relative to the size of the bounding box
        span = max_xy - min_xy
        # Avoid zero span
        span[span < 1e-4] = 1e-4
        
        min_xy -= padding * span
        max_xy += padding * span
        
        # Clamp to [0, 1]
        min_xy = np.maximum(0, min_xy)
        max_xy = np.minimum(1, max_xy)

        x_steps = max(2, int(density * (max_xy[0] - min_xy[0])))
        y_steps = max(2, int(density * (max_xy[1] - min_xy[1])))

        x = np.linspace(min_xy[0], max_xy[0], x_steps)
        y = np.linspace(min_xy[1], max_xy[1], y_steps)

    xv, yv = np.meshgrid(x, y)
    grid = np.stack([xv.flatten(), yv.flatten()], axis=1)
    return grid

def nms_candidates(candidates, scores, threshold=0.05):
    """
    Non-Maximum Suppression for point candidates based on distance.
    """
    if len(candidates) == 0:
        return [], []
        
    # Sort by score descending
    indices = np.argsort(scores)[::-1]
    candidates = candidates[indices]
    scores = scores[indices]
    
    keep = []
    
    while len(candidates) > 0:
        # Pick best
        current = candidates[0]
        current_score = scores[0]
        keep.append((current, current_score))
        
        if len(candidates) == 1:
            break
            
        # Compute distances to rest
        dists = np.linalg.norm(candidates[1:] - current, axis=1)
        
        # Keep only those far enough
        mask = dists > threshold
        candidates = candidates[1:][mask]
        scores = scores[1:][mask]
        
    return [k[0] for k in keep], [k[1] for k in keep]

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Output Directory
    output_dir = setup_run_directories(CONFIG, 'inference')
    
    # 2. Locate Model
    train_run_name = CONFIG['inference']['model_train_run']
    model_path = os.path.join(CONFIG['paths']['output_root'], "training", train_run_name, "model.pth")
    
    print(f"Loading model from: {model_path}")
    print(f"Saving results to: {output_dir}")

    # Load Model
    model = BoltCompletionGNN(
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers']
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Paths
    input_dir = CONFIG['paths']['val_labels'] # Or any inference folder
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for filename in tqdm(files, desc="Inference"):
        # Load existing labels
        labels = load_yolo_labels(os.path.join(input_dir, filename))
        
        if len(labels) == 0:
            continue
            
        # Filter: Use only existing bolts (Class 0) for input
        # Note: In the prompt, Class 1 is missing. We want to predict Class 1.
        # So we take Class 0 as input.
        existing_bolts = labels[labels[:, 0] == 0]
        
        if len(existing_bolts) < 3:
            # Too few points to form a meaningful structure
            save_yolo_labels(os.path.join(output_dir, filename), existing_bolts)
            continue
            
        points = existing_bolts[:, 1:3] # x, y
        
        norm_points = points # Use YOLO coordinates directly

        # Build Graph
        graph = build_graph_from_points(torch.tensor(norm_points, dtype=torch.float32), k=CONFIG['model']['k_neighbors'])
        graph = graph.to(device)
        
        # Generate Grid Candidates
        # We scan the area around the visible points
        grid_candidates = generate_grid_candidates(norm_points, density=CONFIG['inference']['grid_density'])
        cand_tensor = torch.tensor(grid_candidates, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(graph, cand_tensor, k=CONFIG['model']['k_neighbors'])
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        # Filter by probability
        high_prob_mask = probs > CONFIG['inference']['prob_threshold']
        candidates = grid_candidates[high_prob_mask]
        scores = probs[high_prob_mask]
        
        # NMS to find peaks
        final_candidates, final_scores = nms_candidates(candidates, scores, threshold=CONFIG['inference']['nms_threshold'])
        
        if len(final_candidates) > 0:
            final_candidates = np.array(final_candidates)
            abs_candidates = final_candidates # Already in absolute [0,1] coordinates
            
            # Create new labels for predicted missing bolts (Class 1)
            new_labels = []
            for pt in abs_candidates:
                # class 1, x, y, w, h (w,h dummy 0.05)
                new_labels.append([1, pt[0], pt[1], 0.05, 0.05])
            
            new_labels = np.array(new_labels)
            
            # Combine with original existing bolts
            combined_labels = np.vstack([existing_bolts, new_labels])
        else:
            combined_labels = existing_bolts
            
        # Save
        save_yolo_labels(os.path.join(output_dir, filename), combined_labels)

if __name__ == "__main__":
    run_inference()