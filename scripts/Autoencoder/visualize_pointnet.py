import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import read_yolo_label_txt, preds_to_grid, img_read
from model import AE
from config import CHECKPOINT_PATH, GRID_SIZE, DEVICE
import torch


def plot_for(image_path, txt_path, model_path=None, neigh=1):
    items = read_yolo_label_txt(txt_path)
    grid = preds_to_grid(items, GRID_SIZE)
    net = AE(GRID_SIZE).to(DEVICE)
    ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    net.load_state_dict(ck['model_state'])
    net.eval()
    with torch.no_grad():
        inp = torch.from_numpy(grid[None,None,:,:]).float().to(DEVICE)
        out = net(inp).cpu().numpy()[0,0]
    residuals = np.abs(out - grid)

    # compute box scores
    def box_score(it):
        gx = int(np.clip(it['x'] * GRID_SIZE, 0, GRID_SIZE-1))
        gy = int(np.clip(it['y'] * GRID_SIZE, 0, GRID_SIZE-1))
        xs = slice(max(0, gx-neigh), min(GRID_SIZE, gx+neigh+1))
        ys = slice(max(0, gy-neigh), min(GRID_SIZE, gy+neigh+1))
        return float(residuals[ys, xs].mean())

    scores = [box_score(it) for it in items]
    # order boxes
    # show image with boxes
    img = img_read(image_path)
    fig, axs = plt.subplots(1,4, figsize=(20,6))
    axs[0].imshow(img)
    axs[0].set_title('Image')
    W,H = img.size
    for i,it in enumerate(items):
        x=it['x']; y=it['y']; w=it['w']; h=it['h']
        left = (x - w/2) * W
        top = (y - h/2) * H
        right = (x + w/2) * W
        bottom = (y + h/2) * H
        axs[0].add_patch(plt.Rectangle((left, top), right-left, bottom-top, fill=False, edgecolor='yellow', linewidth=2))
        axs[0].text(left, top, f"{scores[i]:.3f}", color='red')

    axs[1].imshow(grid, cmap='gray')
    axs[1].set_title('Input Grid')
    axs[2].imshow(out, cmap='gray')
    axs[2].set_title('Reconstruction')
    axs[3].imshow(residuals, cmap='hot')
    axs[3].set_title('Residuals')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--txt', required=True)
    args = parser.parse_args()
    plot_for(args.image, args.txt)