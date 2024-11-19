import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import random

class PointCloudClassificationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_names = sorted(os.listdir(root_dir))
        print(self.class_names)
        self.samples = []

        # Gather all .pt files along with their labels
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_folder):
                if filename.endswith('.pt'):
                    file_path = os.path.join(class_folder, filename)
                    self.samples.append((file_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_idx = self.samples[idx]
        point_cloud = torch.load(file_path)
        return point_cloud, class_idx

def plot_point_cloud_3d(point_cloud, label, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        point_cloud[:, 0].numpy(), 
        point_cloud[:, 1].numpy(), 
        point_cloud[:, 2].numpy(), 
        c=point_cloud[:, 2].numpy(),  # Color by z-axis
        cmap='viridis', 
        s=1  # Set point size
    )
    ax.set_title(f"Class: {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Save the plot to a file
    plt.savefig(save_path)
    plt.close(fig)


# Set the directory where the dataset is saved
root_dir = "/data/scannet/instances"
dataset = PointCloudClassificationDataset(root_dir)

# Create a DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Plot some samples
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Plot and save a few random samples
for i, (point_cloud, label_idx, target_name, rgb) in enumerate(dataloader):
    if i >= 15:  # Adjust number of samples to plot
        break
    label_name = dataset.class_names[label_idx.item()]
    save_path = os.path.join(output_dir, f"{label_name}_sample_{i}.png")
    plot_point_cloud_3d(point_cloud.squeeze(0), label_name, save_path)
    print(f"Saved plot for {label_name} as {save_path}")
