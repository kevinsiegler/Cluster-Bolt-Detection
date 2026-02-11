import torch
from torch.utils.data import DataLoader
from dataset_pointnet import get_datasets
from model_pointnet import PointNetAE, chamfer_distance
from config import *
from tqdm import tqdm


def train():
    train_ds, val_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE) if val_ds else None

    model = PointNetAE(LATENT_DIM, MAX_POINTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = 1e9

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for pts, n in tqdm(train_loader):
            pts = pts.to(DEVICE)
            recon = model(pts)
            loss = chamfer_distance(pts, recon)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for pts, n in val_loader:
                    pts = pts.to(DEVICE)
                    recon = model(pts)
                    val_loss += chamfer_distance(pts, recon).item()
            val_loss /= len(val_loader)

            if val_loss < best:
                best = val_loss
                torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch {epoch} TrainLoss={total/len(train_loader):.6f} ValLoss={val_loss:.6f}")


if __name__ == '__main__':
    train()