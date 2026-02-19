r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\data_preparation.py
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import load_yolo_labels, CONFIG

class BoltCompletionDataset(Dataset):
    def __init__(self, label_dir, mode='train'):
        self.label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')]
        self.mode = mode
        self.config = CONFIG['training']

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        # Load labels
        labels = load_yolo_labels(self.label_files[idx])
        
        if len(labels) == 0:
            # Return dummy data if file is empty
            return {
                'visible_pos': torch.zeros((0, 2)),
                'candidate_pos': torch.zeros((0, 2)),
                'candidate_labels': torch.zeros((0, 1))
            }

        # Extract coordinates (ignore class for now, treat all as potential structure)
        # In training, we treat ALL existing points as the "Ground Truth Complete Set"
        # Then we artificially mask some to create the input.
        points = labels[:, 1:3] # x, y
        norm_points = points # The points are already normalized in YOLO format

        # --- AUGMENTATION & MASKING ---
        num_points = len(norm_points)
        if self.mode == 'train' and num_points > 2:
            # Determine how many to mask (remove)
            mask_ratio = np.random.uniform(self.config['mask_ratio_min'], self.config['mask_ratio_max'])
            num_mask = max(1, int(num_points * mask_ratio))
            
            # Random permutation
            perm = np.random.permutation(num_points)
            mask_indices = perm[:num_mask]
            keep_indices = perm[num_mask:]
            
            visible_pts = norm_points[keep_indices]
            missing_pts = norm_points[mask_indices] # These are Positive Candidates
            
            # Add Noise to visible points
            noise = np.random.normal(0, self.config['noise_scale'], visible_pts.shape)
            visible_pts = visible_pts + noise
            
            # Generate Negative Candidates (random points)
            num_neg = int(len(missing_pts) * self.config['neg_pos_ratio'])
            # Ensure we have at least some negatives
            num_neg = max(num_neg, 5)
            
            # Generate negatives within [0, 1]
            neg_candidates = np.random.uniform(0, 1, (num_neg, 2))
            
            # Combine Candidates
            # Positives (Missing) = Label 1
            # Negatives (Random) = Label 0
            cand_pos = np.vstack([missing_pts, neg_candidates])
            cand_labels = np.concatenate([np.ones(len(missing_pts)), np.zeros(len(neg_candidates))])
            
        else:
            # Validation mode: We can't dynamically mask efficiently for consistent metrics here
            # unless we fix the seed. For simplicity, we do a fixed split or just return 
            # the full set as visible and no candidates (evaluation is handled differently in evaluate.py)
            # But to make the collate_fn work, we simulate a simple split.
            visible_pts = norm_points
            cand_pos = np.zeros((0, 2))
            cand_labels = np.zeros((0,))

        return {
            'visible_pos': torch.tensor(visible_pts, dtype=torch.float32),
            'candidate_pos': torch.tensor(cand_pos, dtype=torch.float32),
            'candidate_labels': torch.tensor(cand_labels, dtype=torch.float32).unsqueeze(1)
        }

def collate_fn(batch):
    """
    Custom collate to handle variable number of points.
    Returns a list of dictionaries or a batched object.
    Since we process one graph at a time or need custom batching for PyG,
    we will return a list and handle batching in the training loop or use PyG DataLoader if we convert first.
    Here we return a list of dicts.
    """
    return batch