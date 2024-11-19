import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# NYU40 label-to-color mapping (extend as needed)
NYU40_COLORS = {
    0: (0, 0, 0),
    1: (174, 199, 232),    # wall
    2: (152, 223, 138),    # floor
    3: (31, 119, 180),     # cabinet
    4: (255, 187, 120),    # bed
    5: (188, 189, 34),     # chair
    6: (140, 86, 75),      # sofa
    7: (255, 152, 150),    # table
    8: (214, 39, 40),      # door
    9: (197, 176, 213),    # window
    10: (148, 103, 189),   # bookshelf
    11: (196, 156, 148),   # picture
    12: (23, 190, 207),    # counter
    13: (178, 76, 76),    
    14: (247, 182, 210),   # desk
    15: (66, 188, 102),    
    16: (219, 219, 141),   # curtain
    17: (140, 57, 197),    
    18: (202, 185, 52),    
    19: (51, 176, 203),    
    20: (200, 54, 131),    
    21: (92, 193, 61),     
    22: (78, 71, 183),     
    23: (172, 114, 82),    
    24: (255, 127, 14),    # refrigerator
    25: (91, 163, 138),    
    26: (153, 98, 156),    
    27: (140, 153, 101),   
    28: (158, 218, 229),   # shower curtain
    29: (100, 125, 154),   
    30: (178, 127, 135),   
    31: (120, 185, 128),   
    32: (146, 111, 194),   
    33: (44, 160, 44),     # toilet
    34: (112, 128, 144),   # sink
    35: (96, 207, 209),    
    36: (227, 119, 194),   # bathtub
    37: (213, 92, 176),    
    38: (94, 106, 211),    
    39: (82, 84, 163),     # otherfurn
    40: (100, 85, 144)
}

# Load the .ply file as a Trimesh object
mesh = trimesh.load("/data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply")

# Extract point cloud vertices and labels
points = mesh.vertices
labels = mesh.metadata['_ply_raw']['vertex']['data']['label']

# Define the classes to visualize
target_classes = {
    "bed": 4,
    "cabinet": 3,
    "chair": 5,
    "sofa": 6,
    "table": 7,
    "door": 8,
    "window": 9,
    "bookshelf": 10,
    "picture": 11,
    "counter": 12,
    "desk": 14,
    "curtain": 16,
    "refrigerator": 24,
    "bathtub": 36,
    "shower curtain": 28,
    "toilet": 33,
    "sink": 34
}

# Assign colors based on labels, set to white for non-target classes
colors = np.zeros((points.shape[0], 3))  # Default to black
for i, label in enumerate(labels):
    if label in target_classes.values():
        colors[i] = np.array(NYU40_COLORS.get(label, [0, 0, 0])) / 255.0  # Normalize to [0, 1]
    else:
        colors[i] = [1, 1, 1]  # Set other points to white

# Create a 3D scatter plot for target classes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)

# Set equal scaling for all axes
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

# Manually adjust the limits of the axes for equal scaling
max_range = np.array([points[:, 0].max() - points[:, 0].min(), 
                      points[:, 1].max() - points[:, 1].min(), 
                      points[:, 2].max() - points[:, 2].min()]).max() / 2.0

mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.view_init(elev=90, azim=-90) 

# Save the figure
plt.savefig("trimesh_3d_equal_axis_visualization_filtered.png", dpi=300)
plt.close()

print("3D segmentation visualization (filtered) saved as trimesh_3d_equal_axis_visualization_filtered.png")

# Load the .ply file as a Trimesh object
mesh = trimesh.load("/data/scannet/scans/scene0000_00/scene0000_00_vh_clean.ply")


# Extract point cloud vertices and labels
points = mesh.vertices

# Load the aggregation file
with open("/data/scannet/scans/scene0000_00/scene0000_00_vh_clean.aggregation.json", 'r') as f:
    aggregation_data = json.load(f)

# Extract the object instance IDs and their corresponding labels
object_instances = aggregation_data['segGroups']

# Load over-segmentation data
with open("/data/scannet/scans/scene0000_00/scene0000_00_vh_clean.segs.json", 'r') as f:
    seg_data = json.load(f)

# Extract segmentation mappings
seg_to_point_map = seg_data['segIndices']


for i, instance in enumerate(object_instances):
    instance_id = instance['id']
    label_id = instance['label']  # This is the NYU40 label
    if label_id.lower() not in target_classes:
        continue

    # Get segments corresponding to this object instance
    segments = instance['segments']
    # Extract the points for this instance from the point cloud
    instance_points = []
    for seg in segments:
        seg_indices = np.where(np.isin(seg_to_point_map, seg))[0]
        instance_points.append(points[seg_indices])

    instance_points = np.concatenate(instance_points, axis=0)

    # Visualize or save the instance points (mesh or point cloud)
    visualize_object(instance_points, label_id, i)


def visualize_object(points, label, i):
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
    plt.savefig(f"trimesh_3d_equal_axis_visualization_object_{label}_{i}.png", dpi=300)
    plt.close()