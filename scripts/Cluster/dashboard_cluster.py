import streamlit as st
import os
import sys
import cv2
import numpy as np
import yaml
import pickle
import random
from scipy.spatial.distance import cdist

# --- Pfad-Management, um Importe zu ermöglichen ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) # Fügt das übergeordnete 'scripts'-Verzeichnis hinzu
from Cluster.utils import load_config, load_yolo_labels

# --- Hilfsfunktionen ---

def find_image_path(image_id, image_folders):
    for folder in image_folders:
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_boxes(image, boxes, color, thickness):
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

@st.cache_data(show_spinner="Lade und bewerte Inferenz-Ergebnisse (dies dauert einen Moment)...")
def load_and_evaluate_all_data():
    """
    Lädt die Ergebnisse eines abgeschlossenen Inferenz-Laufs und evaluiert sie.
    Diese Funktion wird nur einmal ausgeführt und dient der Filterung der Ergebnisse.
    Dies wird nur einmal ausgeführt und dann gecached.
    """
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    
    # --- Pfade ---
    run_name = cfg['inference']['run_name']
    pred_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    val_gt_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_gt")
    
    if not os.path.isdir(pred_dir):
        st.error(f"Inferenz-Verzeichnis nicht gefunden: `{pred_dir}`. Bitte führen Sie zuerst `inference.py` aus.")
        return []

    # --- Lade alle relevanten Dateien ---
    eval_files = [f for f in os.listdir(val_gt_dir) if f.endswith('.npy')]
    eval_dist_thresh = cfg['evaluation']['dist_threshold']
    
    all_results = []

    for filename in eval_files:
        image_id = os.path.splitext(filename)[0]
        pred_path = os.path.join(pred_dir, f"{image_id}.txt")
        if not os.path.exists(pred_path): continue
        
        # Lade Ground Truth (fehlende Schrauben) und die Prognose
        gt_data = np.load(os.path.join(val_gt_dir, filename))
        pred_labels = load_yolo_labels(pred_path)
        
        candidate_points = pred_labels[pred_labels[:, 0] == 1][:, 1:5] if len(pred_labels) > 0 else np.empty((0, 4))
        
        # Evaluiere die geladene Prognose
        gt_pts_xy = gt_data[:, :2] if gt_data.shape[0] > 0 else np.empty((0, 2))
        pred_pts_xy = candidate_points[:, :2] if candidate_points.shape[0] > 0 else np.empty((0, 2))

        tp, fp, fn = 0, 0, 0
        n_gt, n_pred = gt_pts_xy.shape[0], pred_pts_xy.shape[0]

        if n_gt > 0 and n_pred > 0:
            dists = cdist(gt_pts_xy, pred_pts_xy)
            matched_gt_indices = set()
            matched_pred_indices = set()
            for i in range(n_gt):
                best_pred_idx = np.argmin(dists[i])
                min_dist = dists[i, best_pred_idx]
                if min_dist < eval_dist_thresh and best_pred_idx not in matched_pred_indices:
                    matched_pred_indices.add(best_pred_idx)
                    matched_gt_indices.add(i)
            tp = len(matched_gt_indices)
            fn = n_gt - tp
            fp = n_pred - tp
        elif n_gt > 0:
            fn = n_gt
        elif n_pred > 0:
            fp = n_pred

        all_results.append({
            "image_id": image_id,
            "tp": tp, "fp": fp, "fn": fn
        })
    return all_results

def compute_visualization_data(image_id, cfg, prototypes):
    """Berechnet die Visualisierungsdaten für ein einzelnes Bild bei Bedarf."""
    val_input_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_input")
    val_gt_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_gt")
    pred_dir = os.path.join(cfg['paths']['output_root'], "inference", cfg['inference']['run_name'])

    input_data = np.load(os.path.join(val_input_dir, f"{image_id}.npy"))
    gt_data = np.load(os.path.join(val_gt_dir, f"{image_id}.npy"))
    gt_missing_boxes = gt_data if gt_data.ndim == 2 and gt_data.shape[0] > 0 else np.empty((0, 4))

    pred_labels = load_yolo_labels(os.path.join(pred_dir, f"{image_id}.txt"))
    candidate_points = pred_labels[pred_labels[:, 0] == 1][:, 1:5] if len(pred_labels) > 0 else np.empty((0, 4))

    best_score = float('inf')
    best_aligned_proto_xy = None
    if input_data.shape[0] >= 2:
        input_pts_xy = input_data[:, :2]
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

    return {
        "input_data": input_data,
        "candidate_points": candidate_points,
        "best_aligned_proto_xy": best_aligned_proto_xy,
        "true_positives": np.array(true_positives),
        "false_positives": np.array(false_positives),
        "false_negatives": np.array(false_negatives),
    }

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("Cluster-Vervollständigung: Analyse-Dashboard")

# Lade alle Ergebnisse (wird gecached)
evaluation_results = load_and_evaluate_all_data()

@st.cache_resource
def load_models_and_configs():
    """Lädt Prototypen und Configs, die für die on-demand Visualisierung benötigt werden."""
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    model_name = cfg['clustering'].get('model_name', 'prototypes')
    model_path = os.path.join(cfg['paths']['output_root'], cfg['paths']['model_dir'], f"{model_name}.pkl")
    with open(model_path, 'rb') as f:
        prototypes = pickle.load(f)
    
    gnn_cfg_path = os.path.join(SCRIPT_DIR, '..', 'GNN', 'config.yaml')
    with open(gnn_cfg_path, 'r') as f:
        gnn_cfg = yaml.safe_load(f)
    image_folders = [gnn_cfg["paths"]["train_images"], gnn_cfg["paths"]["val_images"]]
    return cfg, prototypes, image_folders

cfg, prototypes, image_folders = load_models_and_configs()

# --- Sidebar für Steuerung ---
st.sidebar.header("Einstellungen")
num_images = st.sidebar.slider("Anzahl der Beispiele", 1, 10, 3)
filter_option = st.sidebar.selectbox(
    "Filtere nach Ergebnis-Typ",
    ["Alle", "Perfekte Treffer (TP > 0, FP=0, FN=0)", "Mit falschen Prognosen (FP > 0)", "Mit übersehenen Schrauben (FN > 0)"]
)

if st.sidebar.button("Neue zufällige Bilder laden"):
    # Dieser Button löst einfach einen Re-Run des Skripts aus, was zu einer neuen Zufallsauswahl führt
    pass

# --- Filtere die Ergebnisse basierend auf der Auswahl ---
filtered_results = []
for res in evaluation_results:
    is_perfect = res["tp"] > 0 and res["fp"] == 0 and res["fn"] == 0
    has_fp = res["fp"] > 0
    has_fn = res["fn"] > 0

    if filter_option == "Alle":
        filtered_results.append(res)
    elif filter_option == "Perfekte Treffer (TP > 0, FP=0, FN=0)" and is_perfect:
        filtered_results.append(res)
    elif filter_option == "Mit falschen Prognosen (FP > 0)" and has_fp:
        filtered_results.append(res)
    elif filter_option == "Mit übersehenen Schrauben (FN > 0)" and has_fn:
        filtered_results.append(res)

if not filtered_results:
    st.warning("Keine Bilder für die ausgewählten Kriterien gefunden.")
else:
    # Wähle zufällige Indizes aus der gefilterten Liste
    num_to_sample = min(num_images, len(filtered_results))
    selected_results = random.sample(filtered_results, num_to_sample)

    # --- Hauptanzeige ---
    st.header(f"Zufällige Beispiele für: '{filter_option}'")

    for res in selected_results:
        image_id = res["image_id"]
        
        # --- ON-DEMAND BERECHNUNG FÜR VISUALISIERUNG ---
        with st.spinner(f"Visualisierung für {image_id} wird berechnet..."):
            vis_data = compute_visualization_data(image_id, cfg, prototypes)

        st.divider()
        st.subheader(f"Bild: {image_id}")
        st.write(f"**Ergebnis:** Richtig erkannt (TP): {res['tp']}, Falsche Prognose (FP): {res['fp']}, Übersehen (FN): {res['fn']}")

        img_path = find_image_path(image_id, image_folders)
        if not img_path: continue
        image = cv2.imread(img_path)

        # Dynamische Liniendicke basierend auf Bildbreite für bessere Sichtbarkeit
        base_thickness = max(2, int(image.shape[1] / 720))

        # Erstelle die 3-Panel-Visualisierung on-the-fly
        COLOR_INPUT = (255, 150, 0)      # Hellblau
        COLOR_PROTOTYPE = (255, 0, 255)  # Magenta
        COLOR_TP = (0, 255, 0)           # Grün
        COLOR_FP = (0, 255, 255)         # Gelb
        COLOR_FN = (0, 0, 255)           # Rot
        COLOR_ALIGN_LINE = (0, 255, 0)
        COLOR_TEXT = (255, 255, 255)

        # Panel 1: Input
        img1 = image.copy()
        draw_boxes(img1, vis_data["input_data"][:, :4], COLOR_INPUT, base_thickness + 1)
        cv2.putText(img1, "1. Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

        # Panel 2: Alignment
        img2 = image.copy()
        if vis_data["best_aligned_proto_xy"] is not None:
            avg_w = np.mean(vis_data["input_data"][:, 2]) if vis_data["input_data"].shape[0] > 0 else 0.014
            avg_h = np.mean(vis_data["input_data"][:, 3]) if vis_data["input_data"].shape[0] > 0 else 0.025
            proto_boxes = np.hstack([vis_data["best_aligned_proto_xy"], np.full((vis_data["best_aligned_proto_xy"].shape[0], 2), [avg_w, avg_h])])
            draw_boxes(img2, proto_boxes, COLOR_PROTOTYPE, base_thickness)
        draw_boxes(img2, vis_data["input_data"][:, :4], COLOR_INPUT, base_thickness + 1)
        if vis_data["best_aligned_proto_xy"] is not None:
            h, w = img2.shape[:2]
            dists_align = cdist(vis_data["input_data"][:, :2], vis_data["best_aligned_proto_xy"])
            for i, input_pt in enumerate(vis_data["input_data"][:, :2]):
                match_idx = np.argmin(dists_align[i])
                proto_pt = vis_data["best_aligned_proto_xy"][match_idx]
                p1 = (int(input_pt[0] * w), int(input_pt[1] * h))
                p2 = (int(proto_pt[0] * w), int(proto_pt[1] * h))
                cv2.line(img2, p1, p2, COLOR_ALIGN_LINE, base_thickness, cv2.LINE_AA)
        cv2.putText(img2, "2. Alignment", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

        # Panel 3: Prognose
        img3 = image.copy()
        if 'proto_boxes' in locals() and vis_data["best_aligned_proto_xy"] is not None:
            draw_boxes(img3, proto_boxes, COLOR_PROTOTYPE, base_thickness -1 if base_thickness > 1 else 1)
        draw_boxes(img3, vis_data["input_data"][:, :4], COLOR_INPUT, base_thickness + 1)
        if vis_data["true_positives"].shape[0] > 0:
            draw_boxes(img3, vis_data["true_positives"], COLOR_TP, base_thickness + 2)
        if vis_data["false_positives"].shape[0] > 0:
            draw_boxes(img3, vis_data["false_positives"], COLOR_FP, base_thickness + 2)
        if vis_data["false_negatives"].shape[0] > 0:
            draw_boxes(img3, vis_data["false_negatives"], COLOR_FN, base_thickness + 2)
        cv2.putText(img3, "3. Auswertung der Prognose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(img3, "Korrekt (Gruen), Falsch (Gelb), Uebersehen (Rot)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # Zeige die Bilder in drei Spalten an, um die Auflösung zu erhalten
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img1, channels="BGR", use_container_width=True)
        with col2:
            st.image(img2, channels="BGR", use_container_width=True)
        with col3:
            st.image(img3, channels="BGR", use_container_width=True)