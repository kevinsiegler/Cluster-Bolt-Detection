import os
import sys
import yaml
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_yolo_labels, build_knn_graph
from train_gnn import GNN

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_image_path(image_id, config):
    """
    Finds the image path based on ID.
    Prioritizes the validation folder as per requirements, but checks train as fallback.
    """
    search_paths = [
        config["paths"]["val_images"],
        config["paths"]["train_images"]
    ]
    
    for folder in search_paths:
        if not os.path.exists(folder):
            continue
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_visualization(image, boxes, edge_index, errors, yolo_confs, status, anomaly_thresh, yolo_thresh):
    """
    Draws the complete decision process on the image:
    1. Graph connections (Edges)
    2. Bounding Boxes (Color-coded by decision)
    3. Metrics (YOLO Confidence, GNN Error)
    """
    img_h, img_w = image.shape[:2]
    vis_image = image.copy()
    
    # --- Colors (BGR) ---
    COLOR_KEPT = (0, 255, 0)      # Green
    COLOR_REMOVED = (0, 0, 255)   # Red
    COLOR_EDGE = (0, 255, 255)    # Yellow
    COLOR_TEXT = (0, 0, 0)        # Black Text
    
    # --- 1. Draw Graph Structure (Edges) ---
    # This visualizes the k-NN relationships used by the GNN
    if edge_index is not None:
        src_indices = edge_index[0].cpu().numpy()
        dst_indices = edge_index[1].cpu().numpy()
        
        # Calculate center points for all nodes (boxes)
        centers = []
        for box in boxes:
            xc, yc, w, h = box
            x_px = int(xc * img_w)
            y_px = int(yc * img_h)
            centers.append((x_px, y_px))
            
        # Draw lines between connected nodes
        for s, d in zip(src_indices, dst_indices):
            if s < d: # Draw each edge only once
                cv2.line(vis_image, centers[s], centers[d], COLOR_EDGE, 2, cv2.LINE_AA)

    # --- 2. Draw Bounding Boxes & Decision Metrics ---
    for i, box in enumerate(boxes):
        xc, yc, w, h = box
        error = errors[i]
        conf = yolo_confs[i]
        stat = status[i]
        
        # Convert normalized coordinates to pixels
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        
        # Determine Color based on Status
        color = COLOR_KEPT if stat == "kept" else COLOR_REMOVED
        
        # Draw Box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # --- Explainability Text ---
        # Show YOLO Confidence and GNN Reconstruction Error
        # HC = High Confidence (YOLO), LC = Low Confidence (YOLO)
        conf_type = "HC" if conf >= yolo_thresh else "LC"
        
        text_line1 = f"YOLO: {conf:.2f} ({conf_type})"
        text_line2 = f"GNN Err: {error:.4f}"
        
        # Calculate text size for background box
        font_scale = 0.5
        thickness = 1
        (w1, h1), _ = cv2.getTextSize(text_line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(text_line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        box_w = max(w1, w2) + 6
        box_h = h1 + h2 + 10
        
        # Position text above the box
        text_y = y1 - box_h if y1 - box_h > 0 else y1
        
        # Draw filled background for readability
        cv2.rectangle(vis_image, (x1, text_y), (x1 + box_w, text_y + box_h), color, -1)
        
        # Draw Text
        cv2.putText(vis_image, text_line1, (x1 + 3, text_y + h1 + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)
        cv2.putText(vis_image, text_line2, (x1 + 3, text_y + h1 + h2 + 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

    # --- 3. Legend ---
    legend_x, legend_y = 10, 30
    cv2.putText(vis_image, "Edges: k-NN Graph Structure", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EDGE, 2)
    cv2.putText(vis_image, "Green: Kept (Valid Geometry)", (legend_x, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_KEPT, 2)
    cv2.putText(vis_image, "Red: Removed (Geometric Anomaly)", (legend_x, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_REMOVED, 2)
    cv2.putText(vis_image, f"Anomaly Threshold: {anomaly_thresh}", (legend_x, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return vis_image

def main():
    # 1. Setup & Config
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load GNN Model
    training_run = cfg["inference"]["training_run_to_use"]
    model_path = os.path.join(cfg["paths"]["output_root"], "trained_models", training_run, "model.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print(f"Please check 'inference.training_run_to_use' in config.yaml")
        return

    model = GNN(
        in_channels=cfg["gnn"]["input_features"],
        hidden_channels=cfg["gnn"]["hidden_dim"],
        out_channels=cfg["gnn"]["output_dim"],
        num_layers=cfg["gnn"]["num_layers"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {model_path}")

    # 3. Select Random Files
    label_dir = cfg["paths"]["yolo_inference"]
    all_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    
    if not all_files:
        print(f"❌ No label files found in {label_dir}")
        return
        
    selected_files = random.sample(all_files, min(5, len(all_files)))
    print(f"🔍 Selected {len(selected_files)} random files for visualization.")

    # 4. Prepare Output Directory
    label_folder_name = os.path.basename(os.path.normpath(label_dir))
    output_dir = os.path.join(cfg["paths"]["output_root"], "visualized_GNN", label_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Saving visualizations to: {output_dir}")

    # 5. Process Files
    k = cfg["gnn"]["k_neighbors"]
    yolo_thresh = cfg["inference"]["yolo_confidence_threshold"]
    anomaly_thresh = cfg["inference"]["anomaly_threshold"]

    for filename in tqdm(selected_files, desc="Visualizing GNN Inference"):
        image_id = os.path.splitext(filename)[0]
        label_path = os.path.join(label_dir, filename)
        
        # Load Data
        labels = load_yolo_labels(label_path, with_confidence=True)
        if labels.shape[0] == 0:
            continue
            
        img_path = get_image_path(image_id, cfg)
        if not img_path:
            print(f"⚠️ Image for {image_id} not found.")
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Prepare GNN Input
        boxes = labels[:, 1:5] # x, y, w, h
        gnn_features = labels[:, 1:3] # x, y
        confs = labels[:, 5]
        
        # Build Graph (Full Context)
        graph = build_knn_graph(gnn_features, k=k)
        if graph is None:
            continue
        graph = graph.to(device)
        
        # Run Inference
        with torch.no_grad():
            reconstructed_x = model(graph)
            
        # Calculate Reconstruction Errors
        errors = torch.norm(reconstructed_x - graph.x, p=2, dim=1).cpu().numpy()
        
        # Determine Status (Replicating inference logic)
        status = []
        for i in range(len(labels)):
            conf = confs[i]
            err = errors[i]
            
            if conf >= yolo_thresh:
                status.append("kept") # High confidence YOLO is trusted
            else:
                if err > anomaly_thresh:
                    status.append("removed") # Low conf + High error = Anomaly
                else:
                    status.append("kept") # Low conf + Low error = Validated

        # Draw Visualization
        vis_image = draw_visualization(
            image, boxes, graph.edge_index, errors, confs, status, 
            anomaly_thresh, yolo_thresh
        )
        
        # Save Result
        save_path = os.path.join(output_dir, f"{image_id}_gnn_vis.jpg")
        cv2.imwrite(save_path, vis_image)
        
    print("\n✅ Visualization complete.")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()