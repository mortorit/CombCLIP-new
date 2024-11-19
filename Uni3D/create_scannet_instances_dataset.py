import os
import torch
from torch.utils.data import DataLoader
from multiprocessing import Pool
from tqdm import tqdm
from scannet import ScanNetObjectDataset
from functools import partial

# Function to save point clouds to their respective class folders
def save_point_clouds(scene_data, target_classes, output_dir, scene_idx):
    pcs, target_idxs, _, _ = scene_data  # Unpack data

    for obj_idx, pc in enumerate(pcs):
        target_class = list(target_classes.keys())[target_idxs[obj_idx]]
        class_folder = os.path.join(output_dir, target_class)

        # Create class folder if it doesn't exist
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Name the file according to the specified format
        file_name = f"scene_{scene_idx}_obj_{obj_idx + 1}.pt"  # +1 for 1-based index
        file_path = os.path.join(class_folder, file_name)

        # Save the point cloud as a .pt file
        torch.save(pc, file_path)  # Save as PyTorch tensor

def process_scene(scene_idx, dataset, target_classes, output_dir):
    # Get data for the current scene
    scene_data = dataset[scene_idx]
    save_point_clouds(scene_data, target_classes, output_dir, scene_idx)

def main(root_dir, output_dir, num_workers=4, num_points=10000):
    # Create dataset and dataloader
    dataset = ScanNetObjectDataset(root_dir, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define a target classes mapping
    target_classes = dataset.target_classes

    # Prepare a list of scene indices
    scene_indices = range(len(dataset))

    # Create a pool of workers to process scenes in parallel
    with Pool(num_workers) as pool:
        # Use partial to fix the dataset, target_classes, and output_dir
        process_scene_partial = partial(process_scene, dataset=dataset, target_classes=target_classes, output_dir=output_dir)
        list(tqdm(pool.imap(process_scene_partial, scene_indices), total=len(scene_indices)))


if __name__ == "__main__":
    root_dir = "/data/scannet"  # Update this path
    output_dir = "/data/scannet/instances"       # Update this path

    main(root_dir, output_dir)
