from torch.utils.data import DataLoader
from scannet import ScanNetObjectDataset
import torch

def custom_collate_fn(batch):
    """Custom collate function to handle multiple scenes in a batch."""
    batch_pcs = []         # List to store point clouds from all scenes
    batch_target_idxs = [] # List to store target indices from all scenes
    batch_target_names = []# List to store target names from all scenes
    batch_rgb_values = []  # List to store RGB values from all scenes
    
    for scene in batch:
        pcs, target_idxs, target_names, rgb_values = scene
        
        # Extend the batch lists with the data from each scene
        batch_pcs.extend(pcs)
        batch_target_idxs.extend(target_idxs)
        batch_target_names.extend(target_names)
        batch_rgb_values.extend(rgb_values)

    # Stack the point clouds and RGB values into tensors
    batch_pcs = torch.stack(batch_pcs, dim=0)
    batch_rgb_values = torch.stack(batch_rgb_values, dim=0)
    
    # Convert target indices and names to tensors (target names can stay as a list)
    batch_target_idxs = torch.tensor(batch_target_idxs)
    
    return batch_pcs, batch_target_idxs, batch_target_names, batch_rgb_values

# Example usage with DataLoader:
dataset = ScanNetObjectDataset(root_dir="/data/scannet")
print(dataset[0])
dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

# Loop over the DataLoader
for batch_pcs, batch_target_idxs, batch_target_names, batch_rgb_values in dataloader:
    print("Point Clouds Batch Shape:", batch_pcs.shape)
    print("Targets Indices:", batch_target_idxs)
    print("Target Names:", batch_target_names)
    print("RGB Values Batch Shape:", batch_rgb_values.shape)
