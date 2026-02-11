import cv2
import os
import yaml
import argparse
import numpy as np
from utils import load_yolo_labels

# --- Config ---
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# --- Konstanten ---
MAX_SIZE = 1200  # Maximale Anzeigegröße des Bildes

def find_image_path(image_id):
    """Sucht das Bild in den Train- und Val-Ordnern."""
    for folder in [cfg["paths"]["train_images"], cfg["paths"]["val_images"]]:
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_boxes(image, labels, color, thickness, label_prefix=""):
    """Zeichnet Bounding Boxes auf ein Bild."""
    h, w = image.shape[:2]
    for label in labels:
        # YOLO Format: class, xc, yc, bw, bh, [conf]
        xc, yc, bw, bh = label[1], label[2], label[3], label[4]
        
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Optional: Label mit Konfidenz hinzufügen
        if len(label) > 5:
            conf = label[5]
            text = f"{label_prefix}{int(label[0])} ({conf:.2f})"
        else:
            text = f"{label_prefix}{int(label[0])}"
            
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def main(image_id):
    # 1. Bild laden
    img_path = find_image_path(image_id)
    if not img_path:
        print(f"❌ Bild mit ID '{image_id}' nicht gefunden.")
        return
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Fehler beim Laden des Bildes: {img_path}")
        return

    # 2. Pfade zu den Label-Dateien
    inference_run_name = cfg["inference"]["run_name"] # Lese den aktuellen Inferenz-Lauf
    yolo_pred_path = os.path.join(cfg["paths"]["yolo_inference"], image_id + ".txt")
    gnn_validated_path = os.path.join(cfg["paths"]["output_root"], "validated_labels", inference_run_name, image_id + ".txt")

    # 3. Labels laden
    yolo_labels = load_yolo_labels(yolo_pred_path, with_confidence=True)
    gnn_labels = load_yolo_labels(gnn_validated_path, with_confidence=True)

    print(f"Original YOLO predictions: {len(yolo_labels)} boxes")
    print(f"GNN validated predictions: {len(gnn_labels)} boxes")

    # 4. Boxen identifizieren, die entfernt wurden
    yolo_boxes_set = {tuple(row) for row in yolo_labels}
    gnn_boxes_set = {tuple(row) for row in gnn_labels}
    removed_boxes = np.array(list(yolo_boxes_set - gnn_boxes_set))
    
    # 5. Visualisieren
    vis_image = image.copy()
    thickness = cfg["visualization"]["line_thickness"]
    
    if len(gnn_labels) > 0:
        vis_image = draw_boxes(vis_image, gnn_labels, cfg["visualization"]["kept_box_color"], thickness)
    if len(removed_boxes) > 0:
        vis_image = draw_boxes(vis_image, removed_boxes, cfg["visualization"]["removed_box_color"], thickness)

    h, w = vis_image.shape[:2]
    scale = min(MAX_SIZE / w, MAX_SIZE / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        vis_image = cv2.resize(vis_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow(f"Validation for {image_id}", vis_image)
    print("\nDrücken Sie eine beliebige Taste, um das Fenster zu schließen.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisiert die Ergebnisse der GNN-Validierung.")
    parser.add_argument("--image_id", type=str, required=True, help="Die ID des Bildes (ohne Dateiendung), das visualisiert werden soll.")
    args = parser.parse_args()
    
    main(args.image_id)