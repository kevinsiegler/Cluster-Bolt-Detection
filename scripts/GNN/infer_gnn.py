import os
import glob
import json
import torch
import numpy as np
import argparse

from model import AnomalyGNN
from utils import parse_yolo_labels, build_graph_from_boxes

def infer(config):
    """Haupt-Inferenzfunktion."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {device}")

    # 1. Modell und Konfiguration laden
    model_config_path = os.path.join(config['model_dir'], 'config.json')
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    model = AnomalyGNN(
        in_channels=model_config['in_channels_inference'],
        hidden_channels=model_config['hidden_channels']
    ).to(device)
    
    model_path = os.path.join(config['model_dir'], 'gnn_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modell erfolgreich geladen.")

    # 2. Inferenzdaten verarbeiten
    os.makedirs(config['output_dir'], exist_ok=True)
    inference_files = glob.glob(os.path.join(config['input_dir'], '*.txt'))

    for file_path in inference_files:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(config['output_dir'], base_name)

        # Inferenzdaten haben 6 Spalten inkl. Konfidenz
        boxes_np = parse_yolo_labels(file_path, is_inference=True)
        
        if boxes_np.size == 0:
            print(f"Keine Bounding Boxes in {base_name} gefunden. Leere Datei wird erstellt.")
            open(output_path, 'w').close()
            continue

        boxes_tensor = torch.from_numpy(boxes_np).float()

        # 3. Graph erstellen und Inferenz durchführen
        graph = build_graph_from_boxes(boxes_tensor, k=model_config['k'], is_inference=True).to(device)
        
        with torch.no_grad():
            logits = model(graph)
            # Sigmoid anwenden, um Wahrscheinlichkeiten zu erhalten
            probabilities = torch.sigmoid(logits)

        # 4. Ergebnisse filtern und speichern
        plausible_indices = probabilities >= config['threshold']
        
        # Bounding Boxes behalten, die als plausibel eingestuft wurden
        cleaned_boxes_np = boxes_np[plausible_indices.cpu().numpy()]

        # Bereinigte YOLO-Datei speichern
        # Format: class_id x_center y_center width height confidence
        np.savetxt(output_path, cleaned_boxes_np, fmt='%d %f %f %f %f %f')
        
        num_original = len(boxes_np)
        num_cleaned = len(cleaned_boxes_np)
        print(f"Datei '{base_name}': {num_original} -> {num_cleaned} Bounding Boxes (Schwellenwert: {config['threshold']})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN-Inferenz zur Filterung von YOLO-Ergebnissen.")
    parser.add_argument('--input_dir', type=str, default=r'C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\YOLO\infer\evaluations_w_confidence_txt_data\infer_train_30_epoch_conf(0.01)', help="Pfad zum Ordner mit den YOLO-Inferenzlabels (Output von YOLO).")
    parser.add_argument('--output_dir', type=str, default='outputs/cleaned_labels', help="Verzeichnis zum Speichern der bereinigten Labels (im GNN-Ordner). Standard: outputs/cleaned_labels")
    parser.add_argument('--model_dir', type=str, default='trained_models', help="Verzeichnis mit dem trainierten Modell (im GNN-Ordner). Standard: trained_models")
    parser.add_argument('--threshold', type=float, default=0.5, help="Plausibilitäts-Schwellenwert (0-1). Boxen unter diesem Wert werden entfernt.")
    
    args = parser.parse_args()
    
    config = vars(args)
    infer(config)
