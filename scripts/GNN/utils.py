import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from typing import List, Tuple

def parse_yolo_labels(file_path: str, is_inference: bool = False) -> np.ndarray:
    """
    Liest eine YOLO-Labeldatei und gibt die Bounding-Box-Daten als NumPy-Array zurück.

    Args:
        file_path (str): Der Pfad zur .txt-Datei.
        is_inference (bool): True, wenn die Datei Konfidenzwerte enthält (6 Spalten).
                             False für Trainingsdateien (5 Spalten).

    Returns:
        np.ndarray: Ein Array mit den Bounding-Box-Daten.
                    Format Training: [x, y, w, h]
                    Format Inferenz: [class_id, x, y, w, h, confidence]
                    Gibt ein leeres Array zurück, wenn die Datei leer ist.
    """
    try:
        # Manuelles Einlesen ist robuster gegen Formatierungsprobleme als np.loadtxt
        # und verhindert Warnungen bei leeren Dateien.
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return np.array([])

        # Konvertiere Textzeilen zu Float-Listen
        data_list = [[float(x) for x in line.split()] for line in lines]
        data = np.array(data_list)

        if is_inference:
            # Format: class_id, x, y, w, h, confidence
            return data
        else:
            # Format: class_id, x, y, w, h -> wir ignorieren class_id
            return data[:, 1:] if data.shape[1] >= 5 else np.array([])
    except Exception as e:
        print(f"Fehler beim Parsen der Datei {file_path}: {e}")
        return np.array([])

def build_graph_from_boxes(
    boxes: torch.Tensor, 
    k: int, 
    is_inference: bool = False
) -> Data:
    """
    Erstellt einen Graphen aus Bounding-Box-Daten mithilfe von k-Nearest-Neighbors.

    Args:
        boxes (torch.Tensor): Ein Tensor mit den Bounding-Box-Daten.
        k (int): Die Anzahl der nächsten Nachbarn für den k-NN-Graphen.
        is_inference (bool): True, um die passenden Node-Features für die Inferenz zu verwenden.

    Returns:
        torch_geometric.data.Data: Das Graph-Datenobjekt.
    """
    if boxes.numel() == 0:
        # Behandelt den Fall leerer Inputs
        feature_dim = 5 if is_inference else 4
        return Data(x=torch.empty(0, feature_dim), edge_index=torch.empty(2, 0, dtype=torch.long))

    # Node-Features definieren
    if is_inference:
        # Inferenz: [x, y, w, h, confidence]
        # Wir nutzen nur [x, y, w, h] (Indizes 1 bis 5), um Konsistenz mit dem Training zu wahren.
        node_features = boxes[:, 1:5]
    else:
        # Training: [x, y, w, h]
        node_features = boxes

    # Kanten basierend auf den (x, y)-Koordinaten der Bounding-Box-Zentren erstellen
    # PyG's knn_graph erwartet einen Node-Feature-Matrix, um die Distanz zu berechnen.
    # Wir verwenden nur die Koordinaten (x, y) für die k-NN-Suche.
    box_centers = node_features[:, :2]
    
    # Sicherstellen, dass k nicht größer ist als die Anzahl der Knoten - 1
    num_nodes = node_features.shape[0]
    actual_k = min(k, num_nodes - 1)

    if actual_k <= 0:
        # Wenn nur ein Knoten vorhanden ist, gibt es keine Kanten
        edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        edge_index = knn_graph(box_centers, k=actual_k, loop=False) # [2]

    graph = Data(x=node_features, edge_index=edge_index)
    
    return graph
