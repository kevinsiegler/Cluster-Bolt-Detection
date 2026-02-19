r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\graph_builder.py
"""
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def build_graph_from_points(points, k=5):
    """
    Constructs a PyG Data object from a set of (x, y) points.
    
    Args:
        points: (N, 2) tensor of normalized coordinates.
        k: Number of neighbors.
        
    Returns:
        data: PyG Data object with x, edge_index, edge_attr.
    """
    num_nodes = points.shape[0]
    
    # Node features: Just the coordinates (or could be constant 1s if we want pure structure)
    # Using coords helps the model understand absolute position within the normalized cluster
    x = points.float()
    
    if num_nodes < 2:
        # Handle edge case with single point
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    else:
        # Dynamic k: cannot have more neighbors than nodes-1
        curr_k = min(k, num_nodes - 1)
        edge_index = knn_graph(x, k=curr_k, loop=False)
        
        # Calculate Edge Features
        row, col = edge_index
        diff = x[col] - x[row] # dx, dy
        dist = torch.norm(diff, p=2, dim=1).unsqueeze(1) # Euclidean distance
        
        # Edge Attr: [dx, dy, dist]
        edge_attr = torch.cat([diff, dist], dim=1)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=points)

def generate_candidates(visible_points, num_negatives=0, bounds=(0, 1)):
    """
    Generates candidate points for training.
    
    Args:
        visible_points: (N, 2) numpy array.
        num_negatives: Number of random negative samples to generate.
        bounds: Tuple (min, max) for generation area.
        
    Returns:
        candidates: (M, 2) numpy array.
    """
    if num_negatives <= 0:
        return np.empty((0, 2))
        
    # Generate random points within the bounding box (0,1 normalized)
    # We assume points are already normalized to roughly [0,1]
    candidates = np.random.uniform(bounds[0], bounds[1], (num_negatives, 2))
    
    return candidates