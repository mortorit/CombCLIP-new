import torch
import numpy as np
from combine import combine_pcs


# load 2 point clouds
pc1 = torch.load("pc1.pt")
pc2 = torch.load("pc2.pt")
pc3 = torch.load("pc3.pt")


# Plot the colored point clouds separately
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r = pc1[3, :] / 255.0  # Normalize RGB values to [0, 1]
g = pc1[4, :] / 255.0
b = pc1[5, :] / 255.0

# Create an array of RGB colors
colors = np.vstack((r, g, b)).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc1[0, :], pc1[1, :], pc1[2, :], c=colors)
#make axes the same length and proportion
ax.set_box_aspect([np.ptp(pc1[0, :]), np.ptp(pc1[1, :]), np.ptp(pc1[2, :])])
plt.show()

r2 = pc2[3, :] / 255.0  # Normalize RGB values to [0, 1]
g2 = pc2[4, :] / 255.0
b2 = pc2[5, :] / 255.0

# Create an array of RGB colors
colors2 = np.vstack((r2, g2, b2)).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc2[0, :], pc2[1, :], pc2[2, :], c=colors2)
# make axes the same length and proportion
ax.set_box_aspect([np.ptp(pc2[0, :]), np.ptp(pc2[1, :]), np.ptp(pc2[2, :])])
plt.show()

r3 = pc3[3, :] / 255.0  # Normalize RGB values to [0, 1]
g3 = pc3[4, :] / 255.0
b3 = pc3[5, :] / 255.0

# Create an array of RGB colors
colors3 = np.vstack((r3, g3, b3)).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc3[0, :], pc3[1, :], pc3[2, :], c=colors3)
# make axes the same length and proportion
ax.set_box_aspect([np.ptp(pc3[0, :]), np.ptp(pc3[1, :]), np.ptp(pc3[2, :])])
plt.show()

# Combine the point clouds
combined_pc = combine_pcs([pc1, pc2, pc3], 10000, 300, "random", [{"size": "normal", "position": (None, None)},
                                                                    {"size": "large", "position": ("over", 2)},
                                                                    {"size": "normal", "position": ("next_to", 0)}])

# Plot the combined point cloud
r = combined_pc[3, :] / 255.0  # Normalize RGB values to [0, 1]
g = combined_pc[4, :] / 255.0
b = combined_pc[5, :] / 255.0

# Create an array of RGB colors
colors = np.vstack((r, g, b)).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(combined_pc[0, :], combined_pc[1, :], combined_pc[2, :], c=colors)
# make axes the same length and proportion
ax.set_box_aspect([np.ptp(combined_pc[0, :]), np.ptp(combined_pc[1, :]), np.ptp(combined_pc[2, :])])
plt.show()







