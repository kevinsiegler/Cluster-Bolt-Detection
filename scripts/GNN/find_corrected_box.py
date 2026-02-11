import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from utils import load_yolo_labels

# --- Config laden ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def find_image_path(image_id, image_folders):
    """Sucht das Bild in den angegebenen Ordnern."""
    for folder in image_folders:
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_boxes(image, labels, color, thickness):
    """Zeichnet Bounding Boxes auf ein Bild."""
    h, w = image.shape[:2]
    if labels.ndim == 1: # Behandelt den Fall, dass nur eine Box vorhanden ist
        labels = labels.reshape(1, -1)
    for label in labels:
        # YOLO Format: class, xc, yc, bw, bh, [conf]
        xc, yc, bw, bh = label[1], label[2], label[3], label[4]
        
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def main():
    """
    Vergleicht originale YOLO-Labels mit GNN-validierten Labels und
    visualisiert die entfernten Bounding Boxes.
    """
    # --- Pfade aus der Konfiguration laden ---
    inference_run_name = cfg["inference"]["run_name"]
    original_labels_dir = cfg["paths"]["yolo_inference"]
    validated_labels_dir = os.path.join(cfg["paths"]["output_root"], "validated_labels", inference_run_name)
    image_source_dirs = [cfg["paths"]["train_images"], cfg["paths"]["val_images"]]
    
    # --- Ausgabeordner für Visualisierungen definieren ---
    output_viz_dir = os.path.join(cfg["paths"]["output_root"], "comparison_visualizations", inference_run_name)
    os.makedirs(output_viz_dir, exist_ok=True)
    
    print(f"Vergleiche originale Labels aus: '{original_labels_dir}'")
    print(f"mit validierten Labels aus Lauf '{inference_run_name}' in: '{validated_labels_dir}'")
    print(f"Speichere Visualisierungen für Lauf '{inference_run_name}' in: '{output_viz_dir}'")
    
    original_files = [f for f in os.listdir(original_labels_dir) if f.endswith(".txt")]
    
    files_with_removed_boxes = 0

    for filename in tqdm(original_files, desc="Vergleiche Dateien"):
        image_id = os.path.splitext(filename)[0]
        
        original_path = os.path.join(original_labels_dir, filename)
        validated_path = os.path.join(validated_labels_dir, filename)
        
        if not os.path.exists(validated_path):
            continue
            
        # Lade beide Label-Dateien
        original_labels = load_yolo_labels(original_path, with_confidence=True)
        validated_labels = load_yolo_labels(validated_path, with_confidence=True)
        
        # Führe die Visualisierung nur durch, wenn Boxen entfernt wurden
        if original_labels.shape[0] > validated_labels.shape[0]:
            files_with_removed_boxes += 1
            
            # Finde die entfernten Boxen über einen Set-Vergleich
            # Konvertiere zu Strings, um Gleitkomma-Ungenauigkeiten zu vermeiden
            original_set = {tuple(map(str, row)) for row in original_labels}
            validated_set = {tuple(map(str, row)) for row in validated_labels}
            removed_set = original_set - validated_set
            
            if not removed_set:
                continue
            
            removed_labels = np.array([list(map(float, row)) for row in removed_set])
            
            # --- Visualisierung ---
            img_path = find_image_path(image_id, image_source_dirs)
            if not img_path:
                print(f"\nWarnung: Bild für ID '{image_id}' nicht gefunden.")
                continue
            
            image = cv2.imread(img_path)
            
            # Zeichne behaltene Boxen (grün)
            if validated_labels.shape[0] > 0:
                image = draw_boxes(image, validated_labels, cfg["visualization"]["kept_box_color"], cfg["visualization"]["line_thickness"])
            
            # Zeichne entfernte Boxen (rot)
            image = draw_boxes(image, removed_labels, cfg["visualization"]["removed_box_color"], cfg["visualization"]["line_thickness"])
            
            # Speichere das Ergebnisbild
            output_image_path = os.path.join(output_viz_dir, f"{image_id}.jpg")
            cv2.imwrite(output_image_path, image)

    print(f"\n🎉 Vergleich abgeschlossen.")
    print(f"{files_with_removed_boxes} Dateien mit entfernten Boxen gefunden.")
    print(f"Die Visualisierungen für Lauf '{inference_run_name}' wurden in '{output_viz_dir}' gespeichert.")

if __name__ == "__main__":
    main()