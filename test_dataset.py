import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scannet import ScanNetObjectDataset

# Initialize the dataset
root_dir = "/data/scannet"  # Replace with the actual path to your ScanNet dataset
dataset = ScanNetObjectDataset(root_dir)

# Function to visualize an object instance
def visualize_object(points, label, scene_idx, obj_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points in 3D
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)
    
    # Set equal scaling for all axes
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f"Object with label: {label}")
    plt.savefig(f"trimesh_3d_equal_axis_visualization_scene_{scene_idx}_object_{label}_{obj_idx}.png", dpi=300)
    plt.close()
    print(f"Saved visualization for object {obj_idx} in scene {scene_idx} with label: {label}")

# Test the dataset and visualize the first few objects from each scene
num_scenes_to_visualize = 5  # Number of scenes to process
for scene_idx in range(num_scenes_to_visualize):
    scene_objects = dataset[scene_idx]
    print(f"Processing scene {scene_idx} with {len(scene_objects)} objects")

    for obj_idx, (points, label) in enumerate(scene_objects):
        print(f"Visualizing object {obj_idx} with label: {label} in scene {scene_idx}")
        visualize_object(points, label, scene_idx, obj_idx)
