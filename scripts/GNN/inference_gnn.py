import torch
import yaml
import os
import numpy as np
from tqdm import tqdm
from utils import load_yolo_labels, build_knn_graph
from train_gnn import GNN # Annahme: train_gnn.py definiert die GNN-Klasse

def filter_and_save_labels(output_path, original_labels, keep_indices):
    """Speichert die gefilterten Labels in einer neuen Datei."""
    kept_labels = original_labels[keep_indices]
    
    with open(output_path, "w") as f:
        for label in kept_labels:
            # Format: class_id x y w h conf
            line = f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f} {label[5]:.6f}\n"
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
    conf_thresh = cfg["inference"]["yolo_confidence_threshold"]
    anomaly_thresh = cfg["inference"]["anomaly_threshold"]
    k = cfg["gnn"]["k_neighbors"]
    
    yolo_files = [f for f in os.listdir(yolo_preds_dir) if f.endswith(".txt")]

    error_log = [] # Zum Protokollieren der Fehler für die Analyse

    for filename in tqdm(yolo_files, desc="Validating YOLO predictions"):
        label_path = os.path.join(yolo_preds_dir, filename)
        
        # Lade YOLO-Vorhersagen (mit Konfidenz)
        # Format: [class, x, y, w, h, conf]
        labels = load_yolo_labels(label_path, with_confidence=True)

        if labels.shape[0] == 0:
            continue

        # 1. Teile Boxen nach Konfidenz auf
        high_conf_mask = labels[:, 5] >= conf_thresh
        low_conf_mask = ~high_conf_mask
        
        high_conf_indices = np.where(high_conf_mask)[0]
        low_conf_indices_in_original = np.where(low_conf_mask)[0]

        # Wenn alle Boxen hohe Konfidenz haben, gibt es nichts zu prüfen
        if len(low_conf_indices_in_original) == 0:
            output_path = os.path.join(output_dir, filename)
            filter_and_save_labels(output_path, labels, np.arange(len(labels)))
            continue

        # 2. Erstelle Graphen aus ALLEN Boxen für den räumlichen Kontext
        all_boxes = labels[:, 1:5] # Nur Geometrie für den Graphen
        graph = build_knn_graph(all_boxes, k=k)
        
        if graph is None:
            continue

        # 3. Führe GNN-Inferenz auf dem Graphen durch
        graph = graph.to(device)
        with torch.no_grad():
            reconstructed_x = model(graph)
        
        # 4. Berechne Rekonstruktionsfehler für jede Box (Node)
        errors = torch.norm(reconstructed_x - graph.x, p=2, dim=1).cpu().numpy()

        # Logge die Fehler der niedrig-konfidenten Boxen für die Analyse
        image_id = os.path.splitext(filename)[0]
        for i, original_idx in enumerate(low_conf_indices_in_original):
            confidence = labels[original_idx, 5]
            error = errors[original_idx]
            error_log.append({
                "image_id": image_id,
                "confidence": confidence,
                "reconstruction_error": error
            })

        # 5. Identifiziere anomale Boxen UNTER den niedrig-konfidenten
        low_conf_errors = errors[low_conf_indices_in_original]
        is_anomalous_mask = low_conf_errors > anomaly_thresh
        anomalous_low_conf_indices = np.where(is_anomalous_mask)[0]
        indices_to_remove = low_conf_indices_in_original[anomalous_low_conf_indices]
        
        # 6. Bestimme die final zu behaltenden Boxen
        all_indices = set(range(len(labels)))
        indices_to_remove_set = set(indices_to_remove)
        keep_indices = sorted(list(all_indices - indices_to_remove_set))

        # 7. Speichere das gefilterte Ergebnis
        output_path = os.path.join(output_dir, filename)
        filter_and_save_labels(output_path, labels, keep_indices)

    # Speichere das Fehlerprotokoll in einer CSV-Datei zur einfachen Analyse
    if cfg["inference"].get("log_anomaly_errors", False) and error_log:
        import csv
        
        # Erstelle einen dedizierten Ordner für die Logs
        log_dir = os.path.join(cfg["paths"]["output_root"], "anomaly_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Benenne die CSV-Datei nach dem Inferenzlauf
        log_path = os.path.join(log_dir, f"{inference_run_name}_error_log.csv")
        print(f"\nWriting anomaly error log to: {log_path}")
        try:
            with open(log_path, 'w', newline='') as csvfile:
                fieldnames = ["image_id", "confidence", "reconstruction_error"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(error_log)
        except IOError as e:
            print(f"❌ Error writing log file: {e}")

    print("\n🎉 Validation complete.")

if __name__ == "__main__":
    main()
