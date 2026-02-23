import torch
import yaml
import os
import numpy as np
from tqdm import tqdm
from utils import load_yolo_labels, build_knn_graph
from train_gnn import GNN # Annahme: train_gnn.py definiert die GNN-Klasse

def save_yolo_labels_5_cols(output_path, labels):
    """Speichert Labels im 5-spaltigen YOLO-Format (class, x, y, w, h)."""
    with open(output_path, "w") as f:
        if labels.shape[0] == 0:
            return
        for label in labels:
            # Format: class_id x y w h
            line = f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
            f.write(line)

def main():
    # --- Config laden ---
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Modell laden ---
    model = GNN(
        in_channels=cfg["gnn"]["input_features"],
        hidden_channels=cfg["gnn"]["hidden_dim"],
        out_channels=cfg["gnn"]["output_dim"],
        num_layers=cfg["gnn"]["num_layers"]
    ).to(device)

    # Lade das angegebene Modell aus den Trainingsläufen
    training_run_to_use = cfg["inference"]["training_run_to_use"]
    model_path = os.path.join(cfg["paths"]["output_root"], "trained_models", training_run_to_use, "model.pt")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"Please run train_gnn.py with run_name '{training_run_to_use}' first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model '{training_run_to_use}' loaded from {model_path}")

    # --- Pfade für Inferenz vorbereiten ---
    yolo_preds_dir = cfg["paths"]["yolo_inference"]
    inference_run_name = cfg["inference"]["run_name"]
    output_dir = os.path.join(cfg["paths"]["output_root"], "validated_labels", inference_run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Filtered labels for run '{inference_run_name}' will be saved to: {output_dir}")

    # --- Inferenz & Anomalie-Erkennung ---
    anomaly_thresh = cfg["inference"]["anomaly_threshold"]
    k = cfg["gnn"]["k_neighbors"]
    
    yolo_files = [f for f in os.listdir(yolo_preds_dir) if f.endswith(".txt")]

    error_log = [] # Zum Protokollieren der Fehler für die Analyse

    for filename in tqdm(yolo_files, desc="Validating YOLO predictions"):
        label_path = os.path.join(yolo_preds_dir, filename)
        
        # Lade YOLO-Vorhersagen (mit Konfidenz)
        # Format: [class, x, y, w, h] -> Output vom Cluster-Skript
        labels = load_yolo_labels(label_path, with_confidence=False)

        if labels.shape[0] == 0:
            continue

        # 1. Teile Boxen nach Klasse auf: 0=Original, 1=Kandidat vom Cluster-Modul
        original_mask = labels[:, 0] == 0
        candidate_mask = labels[:, 0] == 1
        
        original_labels = labels[original_mask]
        candidate_labels = labels[candidate_mask]

        # Wenn keine Kandidaten da sind, gibt es nichts zu prüfen
        if candidate_labels.shape[0] == 0:
            output_path = os.path.join(output_dir, filename)
            save_yolo_labels_5_cols(output_path, original_labels)
            continue

        # 2. Erstelle Graphen aus ALLEN Boxen für den räumlichen Kontext
        gnn_features = labels[:, 1:3] # Nur x, y für den Graphen
        graph = build_knn_graph(gnn_features, k=k)
        
        if graph is None:
            # Sollte nicht passieren, wenn labels.shape[0] > 0
            continue

        # 3. Führe GNN-Inferenz auf dem Graphen durch
        graph = graph.to(device)
        with torch.no_grad():
            reconstructed_x = model(graph)

        # 4. Berechne Rekonstruktionsfehler für jede Box (Node)
        errors = torch.norm(reconstructed_x - graph.x, p=2, dim=1).cpu().numpy()

        # 5. Identifiziere anomale Boxen UNTER den Kandidaten (Klasse 1)
        candidate_errors = errors[candidate_mask] # Fehler nur für Kandidaten
        is_valid_mask = candidate_errors <= anomaly_thresh
        
        validated_candidates = candidate_labels[is_valid_mask]
        
        # 6. Kombiniere Originale und validierte Kandidaten
        final_labels = np.vstack([original_labels, validated_candidates]) if validated_candidates.shape[0] > 0 else original_labels

        # 7. Speichere das bereinigte Ergebnis
        output_path = os.path.join(output_dir, filename)
        save_yolo_labels_5_cols(output_path, final_labels)

    print("\n🎉 Validation complete.")

if __name__ == "__main__":
    main()
