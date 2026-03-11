r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\dashboard_one_shot.py
"""
import streamlit as st
import os
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cdist

# --- Konfiguration ---
BASE_DIR = r"C:\Users\Kevin\Clustererkennung\bolt_detection"
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images", "train")
MODEL_PATH = r"C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\Outputs\models\multi_cluster_prototypes.pkl"

# Die Liste der zu verarbeitenden Bilder
TARGET_IMAGES = [
    "68a3bc3b9adc8ef68d6d31dd",
    "68a4a9b5a314900b6f473422",
    "68a4a9cea314900b6f474a3f",
    "68a4a551138bb651ff69f4bb",
    "68a4a551138bb651ff69f4bc",
    "68a4a551138bb651ff69f4c6",
    "68a4a551138bb651ff69f2f8",
    "68a4a551138bb651ff69f3d9",
    "68cd913cfd570809dde1b4b6",
    "68cd913cfd570809dde1b48b",
    "6764b34fd9f572ca96e54b8c",
    "6764b34fd9f572ca96e54b99"
]

# --- NEU: Parameter für das Matching und die Qualitätskontrolle ---
MATCH_THRESHOLD = 0.02  # Maximale Distanz (normalisiert), damit ein Punkt als Treffer gilt
# Schwellenwerte, um zu entscheiden, ob ein Match "gut genug" ist
MIN_INLIERS_RATIO = 0.6 # Mindestens 60% der Schablonen-Punkte müssen passen
MAX_AVG_DIST = 0.01     # Der durchschnittliche Abstand der passenden Punkte darf nicht zu groß sein

# Farben (BGR für OpenCV)
COLOR_CLASS_0 = (0, 255, 0)    # Grün (Vorhanden)
COLOR_CLASS_1 = (0, 255, 255)  # Gelb (Fehlend / Kandidat)
COLOR_CLUSTER = (255, 0, 255)  # Magenta (Die Schablone)
COLOR_REMOVED = (0, 0, 255)    # Rot (Entfernt/Rauschen)

def load_yolo_labels(path):
    if not os.path.exists(path):
        return np.empty((0, 5))
    labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append([float(x) for x in parts[:5]])
    return np.array(labels)

def find_best_alignment(prototype_points, target_points, threshold=0.05):
    """
    Sucht die beste Verschiebung für einen Prototyp auf die Zielpunkte.
    NEU: Berücksichtigt nur Punkte innerhalb des Bildbereichs [0,1] für die Bewertung.
    
    dass die meisten Punkte übereinstimmen (RANSAC-ähnlicher Ansatz).
    """
    if len(target_points) == 0:
        return None, 0, float('inf')

    best_offset = None
    max_inliers = -1
    best_avg_dist = float('inf')

    for p_idx, p_point in enumerate(prototype_points):
        for t_idx, t_point in enumerate(target_points):
            # Berechne Verschiebung, wenn p_point auf t_point liegen würde
            offset = t_point - p_point
            
            # Wende Verschiebung an
            shifted_proto = prototype_points + offset
            
            # NEU: Filtere nur Punkte, die innerhalb des Bildes (0-1) liegen
            in_bounds_mask = (shifted_proto[:, 0] >= 0) & (shifted_proto[:, 0] <= 1) & \
                             (shifted_proto[:, 1] >= 0) & (shifted_proto[:, 1] <= 1)
            shifted_proto_in_bounds = shifted_proto[in_bounds_mask]

            # Wenn nach Verschiebung keine Punkte mehr im Bild sind, ist dies eine schlechte Passung
            if len(shifted_proto_in_bounds) == 0:
                continue

            # Berechne Distanzen nur für die sichtbaren Punkte der Schablone
            dists = cdist(shifted_proto_in_bounds, target_points)
            min_dists = np.min(dists, axis=1)
            
            # Zähle Inliers (Punkte, die einen nahen Partner haben)
            # Wichtig: Die Anzahl der Inliers bezieht sich auf die Anzahl der *sichtbaren* Schablonen-Punkte
            inliers = np.sum(min_dists < threshold)
            avg_dist = np.mean(min_dists[min_dists < threshold]) if inliers > 0 else float('inf')
            
            # Kriterium: Mehr Inliers ist besser. Bei Gleichstand entscheidet die geringere Distanz.
            if inliers > max_inliers or (inliers == max_inliers and avg_dist < best_avg_dist):
                max_inliers = inliers
                best_offset = offset
                best_avg_dist = avg_dist

    return best_offset, max_inliers, best_avg_dist

def draw_boxes(img, points, w, h, color, thickness=2, label=None):
    h_img, w_img = img.shape[:2]
    for i, pt in enumerate(points):
        cx, cy = pt[0], pt[1]
        x1 = int((cx - w/2) * w_img)
        y1 = int((cy - h/2) * h_img)
        x2 = int((cx + w/2) * w_img)
        y2 = int((cy + h/2) * h_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    st.set_page_config(layout="wide", page_title="One-Shot Cluster Dashboard")
    st.title("🔩 One-Shot Cluster Matching Dashboard")

    # 1. Modell laden
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modell nicht gefunden unter {MODEL_PATH}. Bitte zuerst das (angepasste) `train_single_prototype.py` Skript ausführen.")
        return

    with open(MODEL_PATH, 'rb') as f:
        # NEU: Lade eine Liste von Prototypen
        all_prototypes = pickle.load(f)
    
    st.sidebar.success(f"{len(all_prototypes)} Cluster-Schablonen geladen.")
    st.sidebar.markdown("---")
    st.sidebar.header("Matching Parameter")
    st.sidebar.info(f"Punkt-Match-Distanz: {MATCH_THRESHOLD}")
    st.sidebar.info(f"Min. Match-Anteil: {MIN_INLIERS_RATIO*100:.0f}%")
    st.sidebar.info(f"Max. Ø-Abstand: {MAX_AVG_DIST}")
    st.sidebar.markdown("---")

    # 2. Durch die Bilder iterieren
    for img_id in TARGET_IMAGES:
        st.markdown(f"---")
        st.subheader(f"Bild ID: {img_id}")

        # Pfade
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            p = os.path.join(IMAGE_DIR, img_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        lbl_path = os.path.join(LABEL_DIR, img_id + ".txt")

        if not img_path or not os.path.exists(lbl_path):
            st.warning(f"Bild oder Label nicht gefunden für {img_id}")
            continue

        # Daten laden
        original_img = cv2.imread(img_path)
        labels = load_yolo_labels(lbl_path) # [class, x, y, w, h]
        
        if len(labels) == 0:
            st.warning("Keine Labels in Datei.")
            continue

        # Input Punkte extrahieren (nur x,y)
        input_points = labels[:, 1:3]
        input_classes = labels[:, 0]

        # --- NEUE LOGIK: BESTEN PROTOTYP FINDEN ---
        best_overall_score = (-1, float('inf')) # (inliers, avg_dist)
        best_overall_proto_data = None
        best_overall_offset = None

        for proto_data in all_prototypes:
            proto_points = proto_data['points']
            offset, num_matches, avg_dist = find_best_alignment(proto_points, input_points, threshold=MATCH_THRESHOLD)
            
            current_score = (num_matches, avg_dist)
            
            # Besserer Score, wenn mehr Inliers, oder bei Gleichstand, wenn der Abstand kleiner ist
            if offset is not None and (current_score[0] > best_overall_score[0] or \
               (current_score[0] == best_overall_score[0] and current_score[1] < best_overall_score[1])):
                best_overall_score = current_score
                best_overall_proto_data = proto_data
                best_overall_offset = offset

        # --- NEU: "GUT GENUG" CHECK ---
        is_match_good = False
        if best_overall_proto_data is not None:
            proto_point_count = len(best_overall_proto_data['points'])
            # Wir müssen die Inliers gegen die *sichtbaren* Punkte des besten Alignments prüfen
            aligned_for_check = best_overall_proto_data['points'] + best_overall_offset
            in_bounds_count = np.sum((aligned_for_check[:, 0] >= 0) & (aligned_for_check[:, 0] <= 1) & \
                                     (aligned_for_check[:, 1] >= 0) & (aligned_for_check[:, 1] <= 1))

            inlier_ratio = best_overall_score[0] / in_bounds_count if in_bounds_count > 0 else 0
            avg_dist = best_overall_score[1]
            
            if inlier_ratio >= MIN_INLIERS_RATIO and avg_dist <= MAX_AVG_DIST:
                is_match_good = True

        # --- LOGIK: FINALES ERGEBNIS BERECHNEN ---

        final_points = []
        final_classes = []

        if is_match_good:
            proto_points = best_overall_proto_data['points']
            offset = best_overall_offset
            aligned_proto = proto_points + offset
            
            dists = cdist(aligned_proto, input_points)
            
            for i in range(len(aligned_proto)):
                proto_pt = aligned_proto[i]
                
                # NEU: Ignoriere Punkte außerhalb des Bildes
                if not (0 <= proto_pt[0] <= 1 and 0 <= proto_pt[1] <= 1):
                    continue
                
                min_dist_idx = np.argmin(dists[i])
                min_dist = dists[i, min_dist_idx]
                
                if min_dist < MATCH_THRESHOLD:
                    # Treffer! Position wird von Schablone übernommen, Klasse ist 0 (vorhanden)
                    final_points.append(proto_pt)
                    final_classes.append(0)
                else:
                    # Kein Treffer -> Fehlende Schraube, Klasse ist 1
                    final_points.append(proto_pt)
                    final_classes.append(1)
        else:
            # Wenn kein guter Match, behalte die originalen Labels
            final_points = input_points
            final_classes = input_classes

        final_points = np.array(final_points)
        final_classes = np.array(final_classes)

        # --- VISUALISIERUNG ---
        col1, col2, col3 = st.columns(3)

        # Bild 1: Input (Raw Labels)
        img1 = original_img.copy()
        # Verwende die durchschnittliche Größe des Prototyps für die Boxen
        avg_w, avg_h = best_overall_proto_data['avg_size'] if best_overall_proto_data else (0.01, 0.01)

        for i, pt in enumerate(input_points):
            cls = int(input_classes[i])
            color = COLOR_CLASS_0 if cls == 0 else COLOR_CLASS_1
            draw_boxes(img1, [pt], avg_w, avg_h, color, thickness=2)
        
        col1.image(img1, channels="BGR", caption="1. Input Labels (Original)")

        # Bild 2: Cluster Overlay (Schablone)
        img2 = original_img.copy()
        # NEU: Zeichne Input exakt wie in Bild 1, damit die Farben erhalten bleiben
        for i, pt in enumerate(input_points):
            cls = int(input_classes[i])
            color = COLOR_CLASS_0 if cls == 0 else COLOR_CLASS_1
            draw_boxes(img2, [pt], avg_w, avg_h, color, thickness=2)

        # NEU: Zeichne immer das beste gefundene Cluster, aber markiere es entsprechend
        if best_overall_offset is not None:
            aligned_proto = best_overall_proto_data['points'] + best_overall_offset
            
            if is_match_good:
                # Guter Match: Magenta, "Match"
                overlay_color = COLOR_CLUSTER
                status_text = f"Match: Proto '{best_overall_proto_data['source_id']}'"
                cv2.putText(img2, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)
                cv2.putText(img2, f"Inliers: {best_overall_score[0]}, AvgDist: {best_overall_score[1]:.4f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
            else:
                # Schlechter Match: Rot, "Bester Versuch (abgelehnt)"
                overlay_color = COLOR_REMOVED # Rot
                status_text = "Bester Versuch (abgelehnt)"
                cv2.putText(img2, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)
                cv2.putText(img2, f"Grund: Ratio/Dist zu schlecht", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)

            # Zeichne das Overlay
            draw_boxes(img2, aligned_proto, avg_w, avg_h, overlay_color, thickness=3)
        else:
            # Gar kein Alignment möglich
            cv2.putText(img2, "Kein Alignment möglich", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        col2.image(img2, channels="BGR", caption="2. Cluster Overlay (Schablone)")

        # Bild 3: Ergebnis (Gefiltert & Ergänzt)
        img3 = original_img.copy()
        if len(final_points) > 0:
            for i, pt in enumerate(final_points):
                cls = int(final_classes[i])
                color = COLOR_CLASS_0 if cls == 0 else COLOR_CLASS_1
                label_txt = "OK" if cls == 0 else "MISSING"
                draw_boxes(img3, [pt], avg_w, avg_h, color, thickness=3, label=label_txt)
        
        caption_3 = "3. Ergebnis (angewendet)" if is_match_good else "3. Ergebnis (Original beibehalten)"
        col3.image(img3, channels="BGR", caption=caption_3)

if __name__ == "__main__":
    main()
