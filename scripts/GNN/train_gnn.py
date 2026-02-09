import os
import glob
import json
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import argparse

from model import AnomalyGNN
from utils import parse_yolo_labels, build_graph_from_boxes

def train(config):
    """Haupt-Trainingsfunktion."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Verwende Gerät: {device}")

    # 1. Daten laden und Graphen erstellen
    label_files = glob.glob(os.path.join(config['data_path'], '*.txt'))
    graphs = []
    for file in label_files:
        # Trainingsdaten haben keine Konfidenzwerte
        boxes_np = parse_yolo_labels(file, is_inference=False)
        if boxes_np.size > 0:
            boxes_tensor = torch.from_numpy(boxes_np).float()
            graph = build_graph_from_boxes(boxes_tensor, k=config['k'], is_inference=False)
            graphs.append(graph)
    
    print(f"{len(graphs)} Graphen aus den Trainingsdaten erstellt.")
    
    # DEBUG: Überprüfung, ob Daten korrekt geladen wurden
    if len(graphs) > 0:
        print(f"DEBUG Check: Erster Graph hat {graphs[0].num_nodes} Knoten.")
        print(f"DEBUG Check: Feature-Shape des ersten Graphen: {graphs[0].x.shape} (Erwartet: [N, 4])")

    # PyG DataLoader für Batching von Graphen
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)

    train_loader = DataLoader(graphs, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    # 2. Modell, Loss und Optimizer initialisieren
    # Im Training haben wir 4 Node-Features: [x, y, w, h]
    model = AnomalyGNN(in_channels=4, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # Da wir nur "gesunde" Beispiele haben, wollen wir, dass das Modell für jeden Knoten eine hohe Plausibilität (nahe 1) ausgibt.
    criterion = torch.nn.BCEWithLogitsLoss()

    # 3. Trainings-Loop
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 1. Positive Samples (Echte Daten -> Label 1)
            logits_pos = model(batch)
            loss_pos = criterion(logits_pos, torch.ones_like(logits_pos))
            
            # 2. Negative Samples (Simulierte Fehler -> Label 0)
            # Wir erzeugen künstliche "falsche" Daten, damit das Modell lernt, was NICHT passt.
            # Sonst lernt es einfach nur, immer "1" auszugeben (Loss = 0).
            batch_neg = batch.clone()
            # Rauschen hinzufügen (z.B. 10% der Bildgröße als Standardabweichung)
            noise = torch.randn_like(batch_neg.x) * 0.1
            batch_neg.x = batch_neg.x + noise
            
            logits_neg = model(batch_neg)
            loss_neg = criterion(logits_neg, torch.zeros_like(logits_neg))
            
            loss = loss_pos + loss_neg
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.6f}")

    # 4. Modell und Konfiguration speichern
    os.makedirs(config['model_dir'], exist_ok=True)
    model_path = os.path.join(config['model_dir'], 'gnn_model.pth')
    config_path = os.path.join(config['model_dir'], 'config.json')

    torch.save(model.state_dict(), model_path)
    # Speichere die Konfiguration, die für die Inferenz benötigt wird
    inference_config = {
        'k': config['k'],
        'in_channels_inference': 4, # Inferenz nutzt nur [x, y, w, h], um zum Training zu passen
        'hidden_channels': 64
    }
    with open(config_path, 'w') as f:
        json.dump(inference_config, f, indent=4)

    print(f"Modell gespeichert unter: {model_path}")
    print(f"Konfiguration gespeichert unter: {config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GNN-Training für Bounding-Box-Anomalieerkennung")
    parser.add_argument('--data_path', type=str, default=r'C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\train', help="Pfad zum Ordner mit den YOLO-Trainingslabels.")
    parser.add_argument('--model_dir', type=str, default='trained_models', help="Verzeichnis zum Speichern des Modells (im GNN-Ordner). Standard: trained_models")
    parser.add_argument('--k', type=int, default=4, help="Anzahl der Nachbarn für den k-NN-Graphen.")
    parser.add_argument('--epochs', type=int, default=50, help="Anzahl der Trainingsepochen.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch-Größe für das Training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Lernrate für den Optimizer.")
    
    args = parser.parse_args()
    
    config = vars(args)
    train(config)
