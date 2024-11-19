from collections import OrderedDict
import math
import time
import torch
import torch.utils.data
import collections
import os
from datetime import datetime

from data.datasets import *
from utils import utils
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import is_master, init_distributed_device, world_info_from_env
from utils.params import parse_args

import open_clip
import trimesh
import models.uni3d as models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scannet import PointCloudClassificationDataset

from tqdm import trange
from torch import nn, optim

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def compute_embedding(clip_model, texts, image, device):
    text_embed_all = []
    with torch.no_grad():
        for i in range(texts.shape[0]):
            text_for_one_sample = texts[i].to(device)
            text_embed = clip_model.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed_all.append(text_embed)

        texts = torch.stack(text_embed_all)

    if image is not None:
        image = clip_model.encode_image(image)
        image = image / image.norm(dim=-1, keepdim=True)
        texts = texts.clone().detach()
        image = image.clone().detach()
        return texts, image

    return texts

def subsample(points, num_samples):
        """Subsample the point cloud to a fixed number of points."""
        if len(points) > num_samples:
            idx = np.random.choice(points.shape[0], num_samples, replace=False)
            return points[idx]
        return points

def normalize_to_unit_sphere(points):
    """Center the point cloud and scale it to fit inside the unit sphere."""
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    # Scale the point cloud to fit within the unit sphere
    furthest_distance = np.max(np.linalg.norm(points, axis=1))
    points /= furthest_distance
    
    return points

# Generate text embeddings
def generate_text_embeddings(clip_model, target_classes, device, tokenizer, args):
    
    text_embeddings = []

    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with torch.no_grad():
        print('=> encoding captions')
        text_features = []
        for l in target_classes:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).to(device=args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

    clip_model.to('cpu')
    return text_features


def generate_uniform_points_in_unit_sphere(N):
    # Initialize an empty list to hold the points
    points = []
    while len(points) < N:
        # Generate random points in a cube [-1, 1] for x, y, z coordinates
        samples = 2 * torch.rand((N, 6)) - 1  # Scale to range [-1, 1]
        
        # Calculate squared distance from the origin for each point
        squared_distances = (samples[:,:3] ** 2).sum(dim=-1)
        
        # Filter points inside the unit sphere (distance <= 1)
        inside_sphere = samples[squared_distances <= 1]
        # Add points to the list, stopping if we reach N points
        points.extend(inside_sphere.tolist())
        points = points[:N]  # Trim to exactly N points if overfilled

    return torch.tensor(points, device='cuda')


def laplacian_regularization(points, k=15):
    """
    Differentiable Laplacian regularization loss to encourage smoothness in point cloud.
    
    Args:
        points (torch.Tensor): Point cloud of shape (N, 6), where first 3 columns are XYZ coordinates.
        k (int): Number of nearest neighbors.
        
    Returns:
        torch.Tensor: Laplacian regularization loss.
    """
    # Extract XYZ coordinates
    xyz = points[:, :3]  # Shape: (N, 3)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(xyz, xyz, p=2)  # Shape: (N, N)

    # Get the indices of the k nearest neighbors (excluding the point itself)
    _, knn_indices = dist_matrix.topk(k=k+1, largest=False)
    
    # Exclude self-distance by selecting from column index 1 onward
    knn_indices = knn_indices[:, 1:]  # Shape: (N, k)
    
    # Gather the k nearest neighbors for each point (batch gather operation)
    neighbors = xyz[knn_indices]  # Shape: (N, k, 3)
    
    # Compute the mean of the neighbors for each point
    mean_neighbors = neighbors.mean(dim=1)  # Shape: (N, 3)
    
    # Compute Laplacian loss as the mean squared distance to the mean of neighbors
    laplacian_loss = torch.norm(xyz - mean_neighbors, dim=1).pow(2).mean()

    return laplacian_loss


def repulsion_loss(points, k=5, sigma=0.01):
    """
    Differentiable repulsion loss to prevent clustering of points.
    
    Args:
        points (torch.Tensor): Point cloud of shape (N, 6), where first 3 columns are XYZ coordinates.
        k (int): Number of nearest neighbors to consider for repulsion.
        sigma (float): Distance threshold below which repulsion is applied.
        
    Returns:
        torch.Tensor: Repulsion loss.
    """
    # Extract XYZ coordinates
    xyz = points[:, :3]  # Shape: (N, 3)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(xyz, xyz, p=2)  # Shape: (N, N)

    # Get the indices and distances of the k nearest neighbors (excluding the point itself)
    knn_distances, knn_indices = dist_matrix.topk(k=k+1, largest=False)
    
    # Exclude self-distance (first column)
    knn_distances = knn_distances[:, 1:]  # Shape: (N, k)
    
    # Calculate repulsion loss
    repulsion_loss = torch.relu(sigma - knn_distances).pow(2).sum()  # Penalize only if distance < sigma

    # Normalize by number of points
    return repulsion_loss / xyz.shape[0]

def main(args):
    args, ds_init = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([args.name, datetime.now().strftime("%Y_%m_%d-%H")])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    print("=> create clip teacher...")
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained)
    tokenizer_clip = open_clip.get_tokenizer(args.clip_model)
    clip_model.to('cpu')
    clip_model.text.to('cuda')

    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.to('cpu')

    texts = tokenizer_clip(['A model of a table']).to(device=args.device, non_blocking=True)
    with torch.no_grad():
        text_features = clip_model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    clip_model.text.to('cpu')

    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')

    pc_dim = (1, 10000, 6)
    learnable_pc = nn.Parameter(generate_uniform_points_in_unit_sphere(10000))

    # Optimization setup
    optimizer = optim.Adam([learnable_pc], lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3400, eta_min=0.05)  # Adjust T_max as needed
    loss_fn = nn.MSELoss()

    # Training loop
    num_steps = 3400  # Modify as needed
    saved_images = []


    for step in trange(num_steps):
        # Select a model to use for this step

        # Forward pass
        pc_features = utils.get_model(model).encode_pc(learnable_pc.unsqueeze(0)).squeeze()
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)  # Normalize
        #cosine_similarity = (pc_features * text_features.to(device)).sum(dim=-1)

        # Compute loss
        #loss = loss_fn(cosine_similarity, torch.tensor([1.0], device=device))
        loss = loss_fn(pc_features, text_features)
        loss_lap =  laplacian_regularization(learnable_pc)
        print(loss, loss_lap)
        loss = loss #+ loss_lap * 0.1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate with cosine annealing

        with torch.no_grad():
            learnable_pc[:,:3] = torch.tensor(normalize_to_unit_sphere(learnable_pc[:,:3].cpu().numpy())).float().cuda()

            if step % 100 == 0:
                points = learnable_pc.cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot the points in 3D
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)
                
                # Set equal scaling for all axes
                max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
                mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
                mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
                mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
  
                plt.savefig(f"step_{step}.png", dpi=300)
                plt.close()


if __name__ == '__main__':
    main(sys.argv[1:])
