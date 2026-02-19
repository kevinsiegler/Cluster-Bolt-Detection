import os
import torch
import yaml
from tqdm import tqdm
from utils import load_yolo_labels, build_knn_graph

# Config laden
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def build_dataset(label_folder, k):
    """
    Erstellt ein Graph-Datensatz aus einem Ordner mit YOLO-Label-Dateien.
    """
    graphs = []
    print(f"Building dataset from: {label_folder}")
    
    file_list = [f for f in os.listdir(label_folder) if f.endswith(".txt")]
    
    for filename in tqdm(file_list, desc="Processing labels"):
        label_path = os.path.join(label_folder, filename)
        
        # Lade Ground Truth Labels (class, x, y, w, h)
        labels = load_yolo_labels(label_path, with_confidence=False)
        
        if labels.shape[0] == 0:
            continue
            
        # Node Features sind nur die Geometrie (x, y)
        # Wir ignorieren die Klasse für das Training des räumlichen Modells
        boxes = labels[:, 1:3]
        
        graph = build_knn_graph(boxes, k=k)
        
        if graph is not None:
            # Füge Metadaten für die Nachverfolgung hinzu
            graph.image_id = os.path.splitext(filename)[0]
            graphs.append(graph)
            
    return graphs

if __name__ == "__main__":
    # Sicherstellen, dass die Ausgabeordner existieren
    output_dir = os.path.join(cfg["paths"]["output_root"], "datasets")
    os.makedirs(output_dir, exist_ok=True)

    # Lade k aus der Konfiguration
    k = cfg["gnn"]["k_neighbors"]

    print("--- Building Training Dataset ---")
    train_graphs = build_dataset(cfg["paths"]["train_labels"], k=k)
    train_save_path = os.path.join(output_dir, "train_graphs.pt")
    torch.save(train_graphs, train_save_path)
    print(f"✅ Training graphs created: {len(train_graphs)} graphs saved to {train_save_path}")

    print("\n--- Building Validation Dataset ---")
    val_graphs = build_dataset(cfg["paths"]["val_labels"], k=k)
    val_save_path = os.path.join(output_dir, "val_graphs.pt")
    torch.save(val_graphs, val_save_path)
    print(f"✅ Validation graphs created: {len(val_graphs)} graphs saved to {val_save_path}")
