import os
import json
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

class PointCloudClassificationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_names = sorted(os.listdir(root_dir))
        self.samples = []

        # Gather all .pt files along with their labels
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_folder):
                if filename.endswith('.pt'):
                    file_path = os.path.join(class_folder, filename)
                    self.samples.append((file_path, class_idx, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, target_id, target_name = self.samples[idx]
        point_cloud = torch.load(file_path)

        # Check if point cloud has fewer than 10,000 points
        num_points = point_cloud.size(0)
        if num_points < 10000:
            # Randomly sample indices and duplicate points to reach 10,000
            indices = torch.randint(0, num_points, (10000 - num_points,))
            additional_points = point_cloud[indices]
            point_cloud = torch.cat([point_cloud, additional_points], dim=0)
        
        # Generate RGB values (all points set to [0.4, 0.4, 0.4])
        rgb = torch.ones_like(point_cloud).float() * 0.4
        point_cloud[:, [1, 2]] = point_cloud[:, [2, 1]]
        return point_cloud, target_id, target_name, rgb


class ScanNetObjectDataset:
    def __init__(self, root_dir, num_points=10000):
        # Root directory where the ScanNet dataset is located
        self.root_dir = root_dir
        self.num_points = num_points  # Number of points to subsample
        
        # Define the target classes with their NYU40 labels
        self.target_classes = {
            "bed": 4, "cabinet": 3, "chair": 5, "sofa": 6, "table": 7, 
            "door": 8, "window": 9, "bookshelf": 10, "picture": 11, 
            "counter": 12, "desk": 14, "curtain": 16, "refrigerator": 24, 
            "bathtub": 36, "shower curtain": 28, "toilet": 33, "sink": 34
        }
        
        # Store scene paths and object counts
        self.scenes = []

        # Parse the scene directories and collect all relevant target objects
        for scene_id in os.listdir(os.path.join(root_dir, "scans")):
            scene_path = os.path.join(root_dir, "scans", scene_id)
            if os.path.isdir(scene_path):
                target_objs = self._collect_scene_objects(scene_path)
                if target_objs:  # Only store scenes with target objects
                    self.scenes.append((scene_path, target_objs))

    def _collect_scene_objects(self, scene_path):
        # Load instance segmentation information from the aggregation file
        scene_id = scene_path.split('/')[-1]
        agg_file = os.path.join(scene_path, scene_id + "_vh_clean.aggregation.json")
        seg_file = os.path.join(scene_path, scene_id + "_vh_clean.segs.json")

        if not os.path.exists(agg_file) or not os.path.exists(seg_file):
            return []

        with open(agg_file, 'r') as f:
            aggregation_data = json.load(f)
        
        # Collect target objects in the scene that belong to target classes
        object_instances = aggregation_data['segGroups']
        target_objs = []

        for i, instance in enumerate(object_instances):
            if instance['label'] in self.target_classes.keys():
                target_objs.append(i)

        return target_objs

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # Get the scene path and target objects in this scene
        scene_path, target_objs = self.scenes[idx]
        return self._load_objects_from_scene(scene_path, target_objs)

    def _load_objects_from_scene(self, scene_path, object_idxs):
        scene_id = scene_path.split('/')[-1]

        # Load the scene's .ply file for the point cloud and .json files for segmentation
        ply_file = os.path.join(scene_path, scene_id + "_vh_clean.ply")
        agg_file = os.path.join(scene_path, scene_id + "_vh_clean.aggregation.json")
        seg_file = os.path.join(scene_path, scene_id + "_vh_clean.segs.json")
        
        mesh = trimesh.load(ply_file)
        points = mesh.vertices
        
        with open(agg_file, 'r') as f:
            aggregation_data = json.load(f)
        object_instances = aggregation_data['segGroups']

        with open(seg_file, 'r') as f:
            seg_data = json.load(f)
        seg_to_point_map = seg_data['segIndices']
        
        # List to store points, labels, target names, and RGB values for each object instance
        pcs = []
        target_idxs = []
        target_names = []
        rgb_values = []

        for object_idx in object_idxs:
            # Extract the object instance corresponding to object_idx
            instance = object_instances[object_idx]
            segments = instance['segments']
            instance_points = []

            for seg in segments:
                seg_indices_o = np.where(np.isin(seg_to_point_map, seg))[0]
                 # Remove indices that are out of bounds
                seg_indices = seg_indices_o[seg_indices_o < len(points)]
                if len(seg_indices_o) > len(seg_indices):
                    print('Removed ', len(seg_indices_o) - len(seg_indices), ' points!')
                instance_points.append(points[seg_indices])

            instance_points = np.concatenate(instance_points, axis=0)
            
            # Subsample the point cloud to the specified number of points
            instance_points = self._subsample(instance_points, self.num_points)
            
            # Center and normalize the point cloud to the unit sphere
            instance_points = self._normalize_to_unit_sphere(instance_points)
            
            # Generate RGB values (all points set to [0.4, 0.4, 0.4])
            rgb_value = torch.ones_like(torch.tensor(instance_points)).float() * 0.4
            
            # Get the label and target name
            label = instance['label']
            target_idx = list(self.target_classes.keys()).index(label)
            target_name = label
            
            # Append points, label, target name, and RGB values to the list
            pcs.append(torch.tensor(instance_points).float())
            target_idxs.append(target_idx)
            target_names.append(target_name)
            rgb_values.append(rgb_value)

        return pcs, target_idxs, target_names, rgb_values

    def _subsample(self, points, num_samples):
        """Subsample the point cloud to a fixed number of points."""
        if len(points) > num_samples:
            idx = np.random.choice(points.shape[0], num_samples, replace=False)
            return points[idx]
        return points

    def _normalize_to_unit_sphere(self, points):
        """Center the point cloud and scale it to fit inside the unit sphere."""
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points -= centroid
        
        # Scale the point cloud to fit within the unit sphere
        furthest_distance = np.max(np.linalg.norm(points, axis=1))
        points /= furthest_distance
        
        return points
