import os
import sys
import cv2
import numpy as np
import yaml
import pickle
import random
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- Pfad-Management, um Importe zu ermöglichen ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) # Fügt das übergeordnete 'scripts'-Verzeichnis hinzu
from Cluster.utils import load_config

def find_image_path(image_id, image_folders):
    """Sucht das Bild in den angegebenen Ordnern."""
    for folder in image_folders:
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_boxes(image, boxes, color, thickness):
    """Zeichnet Bounding Boxes auf ein Bild."""
    h, w = image.shape[:2]
    if boxes.ndim == 1: boxes = boxes.reshape(1, -1)
    for box in boxes:
        xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def main():
    # --- 1. Konfigurationen und Modelle laden ---
    print("Lade Konfigurationen und Modelle...")
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    
    # Lade Cluster-Modell
    model_name = cfg['clustering'].get('model_name', 'prototypes')
    model_path = os.path.join(cfg['paths']['output_root'], cfg['paths']['model_dir'], f"{model_name}.pkl")
    with open(model_path, 'rb') as f:
        prototypes = pickle.load(f)
    print(f"✅ Cluster-Modell '{model_name}.pkl' mit {len(prototypes)} Prototypen geladen.")

    # --- 2. Daten finden ---
    val_input_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_input")
    val_files = [f for f in os.listdir(val_input_dir) if f.endswith('.npy')]
    
    if not val_files:
        print(f"❌ Keine Validierungs-Input-Dateien in {val_input_dir} gefunden.")
        return

    # --- 3. Visualisierungs-Loop ---
    output_dir = os.path.join(cfg['paths']['output_root'], "cluster_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Visualisierungen werden gespeichert in: {output_dir}")

    selected_files = random.sample(val_files, min(5, len(val_files)))

    for filename in tqdm(selected_files, desc="Erstelle Visualisierungen"):
        image_id = os.path.splitext(filename)[0]

        # Lade Bild (benötigt Pfade aus GNN-Config für die Bilder)
        gnn_cfg_path = os.path.join(SCRIPT_DIR, '..', 'GNN', 'config.yaml')
        with open(gnn_cfg_path, 'r') as f:
            gnn_cfg = yaml.safe_load(f)
        img_path = find_image_path(image_id, [gnn_cfg["paths"]["train_images"], gnn_cfg["paths"]["val_images"]])
        if not img_path: continue
        image = cv2.imread(img_path)

        # Lade Ground Truth für die fehlenden Schrauben (für die finale Auswertung)
        val_gt_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_gt")
        gt_data = np.load(os.path.join(val_gt_dir, filename))
        gt_missing_boxes = gt_data if gt_data.ndim == 2 and gt_data.shape[0] > 0 else np.empty((0, 4))

        # Lade bekannte Schraubenpositionen
        input_data = np.load(os.path.join(val_input_dir, filename))
        if input_data.shape[0] < 2: continue
        input_pts_xy = input_data[:, :2]

        # --- Führe die Kernlogik der Inferenz erneut aus, um den besten Prototyp zu finden ---
        best_score = float('inf')
        best_aligned_proto_xy = None
        
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

        # --- Generiere die Prognose aus dem besten Alignment ---
        match_thresh = cfg['inference']['match_threshold']
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

        # --- Evaluiere die Prognose, um TP, FP, FN für die Visualisierung zu finden ---
        eval_dist_thresh = cfg['evaluation']['dist_threshold']
        true_positives = []
        false_positives = []
        false_negatives = []

        pred_boxes = candidate_points
        gt_boxes = gt_missing_boxes

        if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
            dists = cdist(gt_boxes[:, :2], pred_boxes[:, :2])
            matched_gt_indices = set()
            matched_pred_indices = set()

            # Finde True Positives
            for i in range(gt_boxes.shape[0]):
                best_pred_idx = np.argmin(dists[i])
                if dists[i, best_pred_idx] < eval_dist_thresh and best_pred_idx not in matched_pred_indices:
                    true_positives.append(pred_boxes[best_pred_idx])
                    matched_gt_indices.add(i)
                    matched_pred_indices.add(best_pred_idx)
            
            # Finde False Positives (übrige Prognosen)
            for j in range(pred_boxes.shape[0]):
                if j not in matched_pred_indices:
                    false_positives.append(pred_boxes[j])

            # Finde False Negatives (übrige Ground Truths)
            for i in range(gt_boxes.shape[0]):
                if i not in matched_gt_indices:
                    false_negatives.append(gt_boxes[i])

        elif gt_boxes.shape[0] > 0: # Nur GT, keine Prognosen
            false_negatives = list(gt_boxes)
        elif pred_boxes.shape[0] > 0: # Nur Prognosen, kein GT
            false_positives = list(pred_boxes)

        true_positives = np.array(true_positives)
        false_positives = np.array(false_positives)
        false_negatives = np.array(false_negatives)

        # --- Erstelle die 3-Panel-Visualisierung ---
        COLOR_INPUT = (255, 150, 0)      # Hellblau
        COLOR_PROTOTYPE = (255, 0, 255)  # Magenta
        COLOR_ALIGN_LINE = (0, 255, 150) # Türkis
        COLOR_TP = (0, 255, 0)           # Grün
        COLOR_FP = (0, 255, 255)         # Gelb
        COLOR_FN = (0, 0, 255)           # Rot
        COLOR_TEXT = (255, 255, 255)     # Weiß

        # Dynamische Liniendicke basierend auf Bildbreite
        base_thickness = max(2, int(image.shape[1] / 800))

        # Panel 1: Input
        img1 = image.copy()
        draw_boxes(img1, input_data[:, :4], COLOR_INPUT, base_thickness + 1)
        cv2.putText(img1, "1. Input (Bekannte Schrauben)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

        # Panel 2: Alignment
        img2 = image.copy()
        if best_aligned_proto_xy is not None:
            avg_w = np.mean(input_data[:, 2]) if input_data.shape[0] > 0 else 0.014
            avg_h = np.mean(input_data[:, 3]) if input_data.shape[0] > 0 else 0.025
            proto_boxes = np.hstack([best_aligned_proto_xy, np.full((best_aligned_proto_xy.shape[0], 2), [avg_w, avg_h])])
            draw_boxes(img2, proto_boxes, COLOR_PROTOTYPE, base_thickness)
        draw_boxes(img2, input_data[:, :4], COLOR_INPUT, base_thickness + 1)
        if best_aligned_proto_xy is not None:
            h, w = img2.shape[:2]
            dists_align = cdist(input_pts_xy, best_aligned_proto_xy)
            for i, input_pt in enumerate(input_pts_xy):
                match_idx = np.argmin(dists_align[i])
                proto_pt = best_aligned_proto_xy[match_idx]
                p1 = (int(input_pt[0] * w), int(input_pt[1] * h))
                p2 = (int(proto_pt[0] * w), int(proto_pt[1] * h))
                cv2.line(img2, p1, p2, COLOR_ALIGN_LINE, base_thickness, cv2.LINE_AA)
        cv2.putText(img2, "2. Alignment (Bestes Muster)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(img2, "Input (Blau) wird auf Prototyp (Magenta) gematcht", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(img2, f"Match Score: {best_score:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # Panel 3: Prognose
        img3 = image.copy()
        # Zeichne zuerst die komplette Schablone in den Hintergrund
        if 'proto_boxes' in locals():
            draw_boxes(img3, proto_boxes, COLOR_PROTOTYPE, base_thickness - 1 if base_thickness > 1 else 1)
        draw_boxes(img3, input_data[:, :4], COLOR_INPUT, base_thickness + 1)
        # Zeichne die bewerteten Prognosen
        if true_positives.shape[0] > 0:
            draw_boxes(img3, true_positives, COLOR_TP, base_thickness + 2)
        if false_positives.shape[0] > 0:
            draw_boxes(img3, false_positives, COLOR_FP, base_thickness + 2)
        if false_negatives.shape[0] > 0:
            draw_boxes(img3, false_negatives, COLOR_FN, base_thickness + 2)
        cv2.putText(img3, "3. Auswertung der Prognose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(img3, "Korrekt (Gruen), Falsch (Gelb), Uebersehen (Rot)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # Kombinieren und Speichern
        max_h = max(img1.shape[0], img2.shape[0], img3.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * max_h / img1.shape[0]), max_h))
        img2 = cv2.resize(img2, (int(img2.shape[1] * max_h / img2.shape[0]), max_h))
        img3 = cv2.resize(img3, (int(img3.shape[1] * max_h / img3.shape[0]), max_h))
        combined_image = cv2.hconcat([img1, img2, img3])
        
        save_path = os.path.join(output_dir, f"{image_id}_cluster_logic.jpg")
        cv2.imwrite(save_path, combined_image)

    print(f"\n🎉 Visualisierung abgeschlossen. Bilder gespeichert in {output_dir}")

if __name__ == "__main__":
    main()