import cv2
import numpy as np
import torch
import os
import json
import argparse

from model import AnomalyGNN
from utils import parse_yolo_labels, build_graph_from_boxes

# Farben (BGR-Format für OpenCV)
COLOR_PLAUSIBLE = (0, 255, 0)  # Grün
COLOR_UNLIKELY = (0, 0, 255)   # Rot
COLOR_EDGE = (255, 255, 0)     # Cyan

def denormalize_box(box, img_width, img_height):
    """Konvertiert YOLO-Koordinaten zurück in Pixel-Koordinaten."""
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

def visualize_results(config):
    """Erstellt eine visuelle Darstellung der GNN-Filterung."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Modell laden (wie im Inferenzskript)
    model_config_path = os.path.join(config['model_dir'], 'config.json')
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    model = AnomalyGNN(in_channels=model_config['in_channels_inference'], hidden_channels=model_config['hidden_channels']).to(device)
    model_path = os.path.join(config['model_dir'], 'gnn_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Bild und Label laden
    image = cv2.imread(config['image_path'])
    if image is None:
        print(f"Bild konnte nicht geladen werden: {config['image_path']}")
        return
    img_height, img_width, _ = image.shape
    
    # Kopien für die Visualisierung erstellen
    img_before = image.copy()
    img_after = image.copy()

    boxes_np = parse_yolo_labels(config['label_path'], is_inference=True)
    if boxes_np.size == 0:
        print("Keine Bounding Boxes zum Visualisieren gefunden.")
        return

    # 3. GNN-Plausibilitäten berechnen
    boxes_tensor = torch.from_numpy(boxes_np).float()
    graph = build_graph_from_boxes(boxes_tensor, k=model_config['k'], is_inference=True).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(graph)).cpu().numpy()

    # 4. Visualisierungen erstellen
    # Vorher: Alle Boxen in einer neutralen Farbe
    for box_data in boxes_np:
        x1, y1, x2, y2 = denormalize_box(box_data[1:5], img_width, img_height)
        cv2.rectangle(img_before, (x1, y1), (x2, y2), COLOR_PLAUSIBLE, 2)

    # Nachher: Boxen basierend auf Plausibilität einfärben
    for i, box_data in enumerate(boxes_np):
        prob = probabilities[i]
        color = COLOR_PLAUSIBLE if prob >= config['threshold'] else COLOR_UNLIKELY
        
        x1, y1, x2, y2 = denormalize_box(box_data[1:5], img_width, img_height)
        cv2.rectangle(img_after, (x1, y1), (x2, y2), color, 2)
        
        # Konfidenz und Plausibilität anzeigen
        label = f"P: {prob:.2f}"
        cv2.putText(img_after, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Optional: Kanten des Graphen zeichnen
    if config['show_edges']:
        edge_index = graph.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src_node_idx, tgt_node_idx = edge_index[:, i]
            
            # Koordinaten der verbundenen Boxen holen
            box1_center = boxes_np[src_node_idx, 1:3] * [img_width, img_height]
            box2_center = boxes_np[tgt_node_idx, 1:3] * [img_width, img_height]
            
            pt1 = tuple(box1_center.astype(int))
            pt2 = tuple(box2_center.astype(int))
            
            cv2.line(img_after, pt1, pt2, COLOR_EDGE, 1)

    # 5. Ergebnisse anzeigen und speichern
    h_concat = cv2.hconcat([img_before, img_after])
    
    # Titel hinzufügen
    cv2.putText(h_concat, "Vorher: Reiner YOLO-Output", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(h_concat, "Nachher: YOLO + GNN-Filter", (img_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    output_filename = os.path.basename(config['image_path'])
    output_path = os.path.join(config['output_dir'], f"vis_{output_filename}")
    cv2.imwrite(output_path, h_concat)
    print(f"Visualisierung gespeichert unter: {output_path}")

    # Optional: Bild direkt anzeigen
    # cv2.imshow("Vergleich", h_concat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualisierung der GNN-Filterung.")
    parser.add_argument('--image_path', type=str, required=True, help="Pfad zum Originalbild.")
    parser.add_argument('--label_path', type=str, required=True, help="Pfad zur zugehörigen YOLO-Inferenz-Labeldatei.")
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations', help="Verzeichnis zum Speichern der Bilder (im GNN-Ordner). Standard: outputs/visualizations")
    parser.add_argument('--model_dir', type=str, default='trained_models', help="Verzeichnis mit dem trainierten Modell (im GNN-Ordner). Standard: trained_models")
    parser.add_argument('--threshold', type=float, default=0.5, help="Plausibilitäts-Schwellenwert für die Farbkodierung.")
    parser.add_argument('--show_edges', action='store_true', help="Zeichnet die Kanten des Graphen.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args)
    visualize_results(config)
