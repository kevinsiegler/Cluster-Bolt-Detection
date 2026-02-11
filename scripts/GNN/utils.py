import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def load_yolo_labels(label_path, with_confidence=False):
    """
    Liest eine einzelne YOLO-Label-Datei.
    
    Args:
        label_path (str): Pfad zur .txt-Datei.
        with_confidence (bool): Ob die Datei eine Konfidenzspalte enthält.
                                Format: (class, x, y, w, h, conf)
                                Sonst: (class, x, y, w, h)

    Returns:
        np.array: Ein Numpy-Array mit den Label-Daten.
                  Gibt ein leeres Array zurück, wenn die Datei leer ist oder nicht existiert.
    """
    if not os.path.exists(label_path):
        return np.empty((0, 6 if with_confidence else 5))
        
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            data.append(list(map(float, parts)))
            
    if not data:
        return np.empty((0, 6 if with_confidence else 5))

    return np.array(data)


def build_knn_graph(boxes, k):
    """
    Erstellt einen k-NN-Graphen aus einer Liste von Bounding Boxes.

    Args:
        boxes (np.array): Array von Boxen, Shape (N, 4), mit [x, y, w, h].
        k (int): Anzahl der nächsten Nachbarn.

    Returns:
        torch_geometric.data.Data: Ein Graph-Objekt für PyG.
                                   Gibt None zurück, wenn keine Boxen vorhanden sind.
    """
    if boxes.shape[0] == 0:
        return None

    # Node features sind die Box-Koordinaten
    x = torch.tensor(boxes, dtype=torch.float)
    num_nodes = x.shape[0]

    # Kanten mit k-NN erstellen
    # Wir brauchen k Nachbarn, also fragen wir nach k+1, da der Punkt selbst der erste Nachbar ist.
    # Sicherstellen, dass k nicht größer ist als die Anzahl der Punkte.
    n_neighbors = min(k + 1, num_nodes)
    
    # Verwende nur die Mittelpunkte (x, y) für die Nachbarschaftssuche
    positions = boxes[:, :2]
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(positions)
    _, indices = nbrs.kneighbors(positions)
    
    src = []
    dst = []
    # Indizes durchgehen, um Kanten zu erstellen
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]: # Den ersten Nachbarn (sich selbst) überspringen
            src.append(i)
            dst.append(j)
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
