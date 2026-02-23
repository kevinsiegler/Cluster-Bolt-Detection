import os
import sys
import cv2
import numpy as np
import yaml
import pickle
import torch
import random
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- Pfad-Management, um Importe aus beiden Modulen zu ermöglichen ---
# Fügt das übergeordnete 'scripts'-Verzeichnis hinzu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from GNN.utils import build_knn_graph
from GNN.train_gnn import GNN
from Cluster.utils import align_points

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_image_path(image_id, image_folders):
    for folder in image_folders:
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_graph_edges(image, points, edge_index, color, thickness):
    """
    Zeichnet die Kanten eines Graphen auf das Bild.
    """
    h, w = image.shape[:2]
    # Berechne die Mittelpunkte aller Knoten in Pixelkoordinaten
    centers = []
    for pt in points: # points sind (x,y)
        x_px = int(pt[0] * w)
        y_px = int(pt[1] * h)
        centers.append((x_px, y_px))
        
    # Zeichne Linien zwischen verbundenen Knoten
    src_indices = edge_index[0].cpu().numpy()
    dst_indices = edge_index[1].cpu().numpy()
    for s, d in zip(src_indices, dst_indices):
        if s < d: # Zeichne jede Kante nur einmal, um Unordnung zu vermeiden
            cv2.line(image, centers[s], centers[d], color, thickness, cv2.LINE_AA)

def draw_boxes_on_image(image, boxes, color, thickness, text=None, text_color=(255,255,255)):
    h, w = image.shape[:2]
    if boxes.ndim == 1: boxes = boxes.reshape(1, -1)
    for i, box in enumerate(boxes):
        # Box Format: x, y, w, h
        xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if text:
            label = text if not isinstance(text, list) else text[i]
            cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    return image

def main():
    # --- 1. Konfigurationen und Modelle laden ---
    print("Loading configurations and models...")
    cluster_cfg_path = os.path.join(SCRIPT_DIR, "config.yaml")
    gnn_cfg_path = os.path.join(SCRIPT_DIR, '..', 'GNN', 'config.yaml')
    
    cluster_cfg = load_yaml(cluster_cfg_path)
    gnn_cfg = load_yaml(gnn_cfg_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lade Cluster-Modell
    model_name = cluster_cfg['clustering'].get('model_name', 'prototypes')
    cluster_model_path = os.path.join(cluster_cfg['paths']['output_root'], cluster_cfg['paths']['model_dir'], f"{model_name}.pkl")
    with open(cluster_model_path, 'rb') as f:
        prototypes = pickle.load(f)

    # Lade GNN-Modell
    gnn_model = GNN(
        in_channels=gnn_cfg["gnn"]["input_features"],
        hidden_channels=gnn_cfg["gnn"]["hidden_dim"],
        out_channels=gnn_cfg["gnn"]["output_dim"],
        num_layers=gnn_cfg["gnn"]["num_layers"]
    ).to(device)
    training_run = gnn_cfg["inference"]["training_run_to_use"]
    gnn_model_path = os.path.join(gnn_cfg["paths"]["output_root"], "trained_models", training_run, "model.pt")
    gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device))
    gnn_model.eval()
    
    print(f"✅ Models loaded. Using device: {device}")

    # --- 2. Zufälliges Bild auswählen ---
    val_input_dir = os.path.join(cluster_cfg['paths']['output_root'], "preprocessing", "val_input")
    val_files = [f for f in os.listdir(val_input_dir) if f.endswith('.npy')]
    
    if not val_files:
        print(f"❌ No validation input files found in {val_input_dir}")
        return

    # --- 3. Visualisierungs-Loop ---
    output_dir = os.path.join(cluster_cfg['paths']['output_root'], "hybrid_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Visualizations will be saved to: {output_dir}")

    # Wähle 5 zufällige Dateien zur Visualisierung
    selected_files = random.sample(val_files, min(5, len(val_files)))

    for filename in tqdm(selected_files, desc="Generating visualizations"):
        image_id = os.path.splitext(filename)[0]

        # Lade Originalbild
        img_path = find_image_path(image_id, [gnn_cfg["paths"]["train_images"], gnn_cfg["paths"]["val_images"]])
        if not img_path: continue
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        # Lade originale Schraubenpositionen
        input_data = np.load(os.path.join(val_input_dir, filename))
        if input_data.shape[0] < 2: continue
        input_pts_xy = input_data[:, :2]

        # --- STUFE 1: CLUSTER COMPLETION ---
        best_score = float('inf')
        best_aligned_proto_xy = None
        best_proto_full = None

        for proto in prototypes:
            proto_pts_xy = proto['points'][:, :2]
            if len(proto_pts_xy) < len(input_pts_xy): continue
            
            current_proto_best_score = float('inf')
            current_proto_best_aligned = None
            for i in range(len(input_pts_xy)):
                for j in range(len(proto_pts_xy)):
                    t = input_pts_xy[i] - proto_pts_xy[j]
                    candidate_proto = proto_pts_xy + t
                    score = np.mean(np.min(cdist(input_pts_xy, candidate_proto), axis=1))
                    if score < current_proto_best_score:
                        current_proto_best_score = score
                        current_proto_best_aligned = candidate_proto
            
            if current_proto_best_score < best_score:
                best_score = current_proto_best_score
                best_aligned_proto_xy = current_proto_best_aligned
                best_proto_full = proto['points']

        match_thresh = cluster_cfg['inference']['match_threshold']
        candidate_points = []
        if best_aligned_proto_xy is not None and best_score < match_thresh:
            dists = cdist(best_aligned_proto_xy, input_pts_xy)
            min_dists_proto_to_input = np.min(dists, axis=1)
            missing_indices = np.where(min_dists_proto_to_input > match_thresh)[0]
            
            avg_w = np.mean(input_data[:, 2])
            avg_h = np.mean(input_data[:, 3])

            for idx in missing_indices:
                pt_xy = best_aligned_proto_xy[idx]
                if 0 <= pt_xy[0] <= 1 and 0 <= pt_xy[1] <= 1:
                    candidate_points.append([pt_xy[0], pt_xy[1], avg_w, avg_h])
        
        candidate_points = np.array(candidate_points)

        # --- STUFE 2: GNN VALIDATION ---
        kept_candidates = []
        removed_candidates = []
        graph = None
        gnn_features = None
        candidate_errors = np.array([])
        anomaly_thresh = gnn_cfg["inference"]["anomaly_threshold"]

        if candidate_points.shape[0] > 0:
            all_points_for_graph = np.vstack([input_data[:, :4], candidate_points])
            gnn_features = all_points_for_graph[:, :2]
            
            graph = build_knn_graph(gnn_features, k=gnn_cfg["gnn"]["k_neighbors"])
            graph = graph.to(device)

            with torch.no_grad():
                reconstructed_x = gnn_model(graph)
            
            errors = torch.norm(reconstructed_x - graph.x, p=2, dim=1).cpu().numpy()
            
            candidate_errors = errors[len(input_data):]

            for i, cand_point in enumerate(candidate_points):
                if candidate_errors[i] <= anomaly_thresh:
                    kept_candidates.append(cand_point)
                else:
                    removed_candidates.append(cand_point)

        kept_candidates = np.array(kept_candidates)
        removed_candidates = np.array(removed_candidates)

        # --- STUFE 3: VISUALISIERUNG ---
        # Farben (BGR) für bessere Lesbarkeit
        COLOR_ORIGINAL = (255, 150, 0)   # Hellblau
        COLOR_PROTOTYPE = (100, 100, 100) # Grau
        COLOR_CANDIDATE = (0, 255, 255)   # Gelb
        COLOR_KEPT = (0, 255, 0)          # Grün
        COLOR_REMOVED = (0, 0, 255)       # Rot
        COLOR_EDGE = (255, 255, 0)        # Cyan
        COLOR_TEXT = (255, 255, 255)      # Weiß

        # --- Bild 1: Cluster: Pattern Matching ---
        img1 = image.copy()
        # Zeichne das gesamte ausgerichtete Prototyp-Muster im Hintergrund
        if best_aligned_proto_xy is not None:
            avg_w = np.mean(input_data[:, 2]) if input_data.shape[0] > 0 else 0.014
            avg_h = np.mean(input_data[:, 3]) if input_data.shape[0] > 0 else 0.025
            proto_boxes = np.hstack([best_aligned_proto_xy, np.full((best_aligned_proto_xy.shape[0], 2), [avg_w, avg_h])])
            draw_boxes_on_image(img1, proto_boxes, COLOR_PROTOTYPE, 1) # Dünne graue Boxen
        # Zeichne die originalen Punkte darüber
        draw_boxes_on_image(img1, input_data[:, :4], COLOR_ORIGINAL, 2)
        # Zeichne die generierten Kandidaten als Highlight
        if candidate_points.shape[0] > 0:
            draw_boxes_on_image(img1, candidate_points, COLOR_CANDIDATE, 3) # Dickere gelbe Boxen
        cv2.putText(img1, "1. Cluster: Pattern Matching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(img1, "Input (Blau), Pattern (Grau), Kandidaten (Gelb)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # --- Bild 2: GNN: Graph & Fehleranalyse ---
        img2 = image.copy()
        # Zeichne alle Punkte, die den Graphen bilden
        draw_boxes_on_image(img2, input_data[:, :4], COLOR_ORIGINAL, 2)
        if candidate_points.shape[0] > 0:
            draw_boxes_on_image(img2, candidate_points, COLOR_CANDIDATE, 2)
            # Zeichne die Graphenkanten
            if graph is not None:
                draw_graph_edges(img2, gnn_features, graph.edge_index, COLOR_EDGE, 1)
            # Zeichne den Rekonstruktionsfehler für jeden Kandidaten
            h_img, w_img = img2.shape[:2]
            for i, cand_point in enumerate(candidate_points):
                error_text = f"{candidate_errors[i]:.4f}"
                xc, yc, _, _ = cand_point
                x_px = int(xc * w_img)
                y_px = int(yc * h_img)
                cv2.putText(img2, error_text, (x_px + 10, y_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        cv2.putText(img2, "2. GNN: Graph & Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(img2, "Kandidaten (Gelb) mit Rekonstruktionsfehler", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # --- Bild 3: GNN: Finale Entscheidung ---
        img3 = image.copy()
        draw_boxes_on_image(img3, input_data[:, :4], COLOR_ORIGINAL, 2)
        if kept_candidates.shape[0] > 0:
            draw_boxes_on_image(img3, kept_candidates, COLOR_KEPT, 3)
        if removed_candidates.shape[0] > 0:
            draw_boxes_on_image(img3, removed_candidates, COLOR_REMOVED, 3)
        cv2.putText(img3, "3. GNN: Final Decision", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(img3, f"Valide (Gruen), Entfernt (Rot) | Thresh: {anomaly_thresh}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # Kombiniere Bilder
        # Skaliere alle auf die gleiche Höhe, falls sie unterschiedlich sind
        max_h = max(img1.shape[0], img2.shape[0], img3.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * max_h / img1.shape[0]), max_h))
        img2 = cv2.resize(img2, (int(img2.shape[1] * max_h / img2.shape[0]), max_h))
        img3 = cv2.resize(img3, (int(img3.shape[1] * max_h / img3.shape[0]), max_h))

        combined_image = cv2.hconcat([img1, img2, img3])
        
        # Speichern
        save_path = os.path.join(output_dir, f"{image_id}_hybrid_steps.jpg")
        cv2.imwrite(save_path, combined_image)

    print(f"\n🎉 Visualization complete. Images saved to {output_dir}")

if __name__ == "__main__":
    main()