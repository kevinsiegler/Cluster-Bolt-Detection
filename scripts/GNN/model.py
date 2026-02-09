import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SAGEConv, global_mean_pool

class AnomalyGNN(torch.nn.Module):
    """
    Eine GNN-Architektur zur Erkennung von strukturellen Anomalien in Bounding-Box-Layouts.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 1):
        """
        Args:
            in_channels (int): Dimensionalität der Eingangs-Node-Features.
                               4 für Training, 5 für Inferenz.
            hidden_channels (int): Anzahl der Features in den versteckten Layern.
            out_channels (int): Dimensionalität des Outputs pro Node (hier: 1 für Plausibilität).
        """
        super().__init__()
        
        # Graph Convolution Layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Node-Level Klassifikations-Head (MLP)
        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data: 'torch_geometric.data.Data') -> torch.Tensor:
        """
        Führt den Forward-Pass für den Graphen durch.

        Args:
            data (torch_geometric.data.Data): Der Eingabegraph.

        Returns:
            torch.Tensor: Ein Logit-Wert pro Node, der die Plausibilität repräsentiert.
        """
        x, edge_index = data.x, data.edge_index

        # Graph Convolutions anwenden
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Node-Klassifikation durchführen
        node_logits = self.classifier(x)
        
        return node_logits.squeeze(-1)
