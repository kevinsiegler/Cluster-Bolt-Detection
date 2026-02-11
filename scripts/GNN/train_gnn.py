import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import yaml
import os
from tqdm import tqdm

# --- Config laden ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- GNN Autoencoder Modell Definition ---
# Diese Klasse wird auch von inference_gnn.py importiert
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2 for an autoencoder.")

        # Encoder-Teil
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Decoder-Teil (letzte Schicht)
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            # Keine Aktivierungsfunktion auf der letzten Schicht
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

def main():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datensätze laden ---
    dataset_dir = os.path.join(cfg["paths"]["output_root"], "datasets")
    train_graphs_path = os.path.join(dataset_dir, "train_graphs.pt")
    val_graphs_path = os.path.join(dataset_dir, "val_graphs.pt")

    if not os.path.exists(train_graphs_path) or not os.path.exists(val_graphs_path):
        print(f"❌ Error: Graph dataset files not found in '{dataset_dir}'")
        print("Please run dataset_builder.py first.")
        return

    train_graphs = torch.load(train_graphs_path, weights_only=False)
    val_graphs = torch.load(val_graphs_path, weights_only=False)
    print(f"✅ Loaded {len(train_graphs)} training graphs and {len(val_graphs)} validation graphs.")

    train_loader = DataLoader(train_graphs, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg["training"]["batch_size"])

    # --- Modell initialisieren ---
    model = GNN(
        in_channels=cfg["gnn"]["input_features"],
        hidden_channels=cfg["gnn"]["hidden_dim"],
        out_channels=cfg["gnn"]["output_dim"],
        num_layers=cfg["gnn"]["num_layers"]
    ).to(device)
    print("\n--- Model Architecture ---\n", model, "\n--------------------------\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    # --- Trainingsloop ---
    best_val_loss = float('inf')
    print("--- Starting Training ---")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_train_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validierungsloop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Val]  "):
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.x)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} -> Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")

        # --- Modell speichern ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Erstelle einen dedizierten Ordner für diesen Trainingslauf
            training_run_name = cfg["training"]["run_name"]
            model_dir = os.path.join(cfg["paths"]["output_root"], "trained_models", training_run_name)
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, "model.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ New best model for run '{training_run_name}' saved to '{model_save_path}' (Val Loss: {avg_val_loss:.6f})")

    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()
