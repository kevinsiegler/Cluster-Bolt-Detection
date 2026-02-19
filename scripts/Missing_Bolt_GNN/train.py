r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\train.py
"""
import torch
import torch.optim as optim
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import CONFIG, setup_run_directories
from data_preparation import BoltCompletionDataset, collate_fn
from graph_builder import build_graph_from_points
from model import BoltCompletionGNN

def train():
    # Setup output directory: output/training/<run_name>
    output_dir = setup_run_directories(CONFIG, 'training')
    model_save_path = os.path.join(output_dir, "model.pth")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training Run: {CONFIG['training']['run_name']}")
    print(f"Saving to: {output_dir}")

    # Dataset
    train_dataset = BoltCompletionDataset(CONFIG['paths']['train_labels'], mode='train')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['training']['batch_size'], 
                              shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Model
    model = BoltCompletionGNN(
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers']
    ).to(device)

    # --- RESUME LOGIC ---
    if os.path.exists(model_save_path):
        print(f"🔄 Found existing model at {model_save_path}. Loading weights to continue training...")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading weights (Architecture mismatch?): {e}")
            print("   Starting fresh training instead.")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['training']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    
    for epoch in range(CONFIG['training']['epochs']):
        model.train()
        total_loss = 0
        total_bce = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['training']['epochs']}")
        
        for batch in pbar:
            batch_loss = 0
            optimizer.zero_grad()
            
            # Process each image in the batch individually (since graph sizes vary wildly)
            # Alternatively, we could use PyG Batch, but our "Candidates" logic is unique.
            # Gradient accumulation is used to simulate batch size.
            
            valid_samples = 0
            
            for sample in batch:
                visible_pos = sample['visible_pos'].to(device)
                candidate_pos = sample['candidate_pos'].to(device)
                labels = sample['candidate_labels'].to(device)
                
                if visible_pos.size(0) < 2 or candidate_pos.size(0) == 0:
                    continue
                
                # Build Graph
                graph = build_graph_from_points(visible_pos, k=CONFIG['model']['k_neighbors'])
                graph = graph.to(device)
                
                # Forward
                logits = model(graph, candidate_pos, k=CONFIG['model']['k_neighbors'])
                
                # Loss 1: BCE (Existence)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                
                # Loss 2: Chamfer-like regularization (Optional)
                # If a point is predicted positive, it should be close to a real missing point.
                # Since we use explicit labels, BCE covers this. 
                # We add a small regularization to keep predictions confident.
                
                loss = bce_loss
                loss.backward()
                
                batch_loss += loss.item()
                valid_samples += 1
            
            if valid_samples > 0:
                optimizer.step()
                total_loss += batch_loss / valid_samples
                pbar.set_postfix({'loss': batch_loss / valid_samples})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        scheduler.step(avg_loss)
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("Model saved.")
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG['training']['patience']:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()