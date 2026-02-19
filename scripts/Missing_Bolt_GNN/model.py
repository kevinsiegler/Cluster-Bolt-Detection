r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MLP

class BoltCompletionGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, output_dim=1):
        super(BoltCompletionGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # --- ENCODER (Process visible structure) ---
        # Input features: 2 (x, y)
        self.node_encoder = nn.Linear(2, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # GATConv allows weighting neighbors differently
            self.convs.append(GATConv(hidden_dim, hidden_dim, edge_dim=3, add_self_loops=True))
            
        # --- DECODER (Query candidates against encoded structure) ---
        # We will concatenate: 
        # 1. Candidate Coord (2)
        # 2. Aggregated features from k-nearest visible nodes (hidden_dim)
        # 3. Relative vector to nearest nodes (2)
        
        self.query_encoder = nn.Linear(2, hidden_dim)
        
        # Final MLP to predict probability
        # Input: [agg_features, cand_embed, relative_pos_to_nearest]
        self.head = MLP([hidden_dim * 2 + 2, hidden_dim, 64, output_dim],
                        act='relu', norm=None, dropout=0.1)

    def encode_visible(self, x, edge_index, edge_attr):
        """
        Encodes the visible graph.
        """
        x = self.node_encoder(x)
        x = F.relu(x)
        
        for conv in self.convs:
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = x + x_res # Residual connection
            
        return x

    def forward(self, visible_data, candidate_pos, k=4):
        """
        Args:
            visible_data: PyG Batch/Data object of visible nodes.
            candidate_pos: (M, 2) Tensor of candidate coordinates.
            k: Number of neighbors to query for each candidate.
        
        Returns:
            logits: (M, 1) Probability logits for candidates.
        """
        # 1. Encode visible nodes
        # visible_data.x is (N, 2)
        encoded_nodes = self.encode_visible(visible_data.x, visible_data.edge_index, visible_data.edge_attr)
        
        # 2. For each candidate, find k nearest visible nodes
        # We use brute-force distance for simplicity as M and N are small (<100)
        # candidate_pos: (M, 2), visible_data.pos: (N, 2)
        
        # Compute pairwise distance matrix (M, N)
        dist_matrix = torch.cdist(candidate_pos, visible_data.pos)
        
        # Get k nearest neighbors
        # values: (M, k), indices: (M, k)
        k_actual = min(k, visible_data.pos.size(0))
        if k_actual == 0:
            return torch.zeros((candidate_pos.size(0), 1), device=candidate_pos.device)
            
        dist_k, indices_k = dist_matrix.topk(k_actual, dim=1, largest=False)
        
        # 3. Aggregate features
        # Gather node features: (M, k, hidden_dim)
        neighbor_features = encoded_nodes[indices_k]
        
        # Simple aggregation: Mean or Max. Let's use Mean.
        # (M, hidden_dim)
        agg_features = torch.mean(neighbor_features, dim=1)
        
        # 4. Get relative position to the SINGLE nearest neighbor
        # This provides a strong geometric clue.
        nearest_neighbor_pos = visible_data.pos[indices_k[:, 0]] # (M, 2)
        relative_pos_to_nearest = nearest_neighbor_pos - candidate_pos # (M, 2)
        
        # 5. Candidate's own position embedding
        cand_embed = self.query_encoder(candidate_pos)
        
        # 6. Concatenate and Predict
        combined = torch.cat([agg_features, cand_embed, relative_pos_to_nearest], dim=1)
        
        logits = self.head(combined)
        return logits