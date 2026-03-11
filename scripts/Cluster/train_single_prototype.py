r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\train_single_prototype.py
"""
import os
import numpy as np
import pickle

# --- Konfiguration ---
LABEL_DIR = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\train"
# NEU: Liste von IDs, um mehrere Prototypen zu trainieren
TARGET_IDS = [
    "68a4a551138bb651ff69f4ad", # Dein ursprünglicher Prototyp
    "68cd913cfd570809dde1b4b5"  # Ein zweiter Prototyp als Beispiel
]
OUTPUT_MODEL_PATH = r"C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\Outputs\models\multi_cluster_prototypes.pkl"

def load_yolo_labels(path):
    """Lädt YOLO Labels (class, x, y, w, h)."""
    if not os.path.exists(path):
        print(f"Datei nicht gefunden: {path}")
        return np.empty((0, 5))
    
    labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Wir lesen alles als Float
                labels.append([float(x) for x in parts[:5]])
    return np.array(labels)

def main():
    # NEU: Liste für alle trainierten Prototypen
    all_prototypes = []
    print(f"Trainiere {len(TARGET_IDS)} Prototypen...")
    
    for target_id in TARGET_IDS:
        label_path = os.path.join(LABEL_DIR, target_id + ".txt")
        print(f"  -> Lese Trainings-Label: {label_path}")
        
        data = load_yolo_labels(label_path)
        
        if len(data) == 0:
            print(f"      Warnung: Keine Labels für ID {target_id} gefunden. Überspringe.")
            continue

        # Wir nehmen nur x, y Koordinaten (Spalte 1 und 2)
        points = data[:, 1:3]
        
        # Wir speichern auch die durchschnittliche Breite/Höhe für die Visualisierung später
        avg_w = np.mean(data[:, 3])
        avg_h = np.mean(data[:, 4])

        # Zentrieren des Clusters (auf den Mittelpunkt der Punktwolke)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        prototype = {
            "points": centered_points,
            "original_centroid": centroid,
            "avg_size": (avg_w, avg_h),
            "source_id": target_id
        }
        all_prototypes.append(prototype)
        print(f"      ...Prototyp '{target_id}' mit {len(points)} Punkten erstellt.")

    # Ordner erstellen falls nicht existent
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

    if all_prototypes:
        with open(OUTPUT_MODEL_PATH, 'wb') as f:
            pickle.dump(all_prototypes, f)
        
        print(f"\n✅ {len(all_prototypes)} Cluster erfolgreich trainiert und gespeichert unter:")
        print(f"   {OUTPUT_MODEL_PATH}")
    else:
        print("\n❌ Keine Prototypen konnten trainiert werden.")

if __name__ == "__main__":
    main()
