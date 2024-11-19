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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_seed(seed):
    random.seed(seed)                        # Python random module
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed(seed)             # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)         # PyTorch GPU (all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable benchmark mode for reproducibility

# Example usage
set_seed(42)

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


def plot_instance_point_cloud(instance_points, instance_id, label_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(instance_points[:, 0], instance_points[:, 1], instance_points[:, 2], s=1, c='blue')
    ax.set_title(f"Instance ID: {instance_id}, Label: {label_id}")
    ax.set_box_aspect([1, 1, 1])

    # Manually adjust the limits of the axes for equal scaling
    max_range = np.array([
        instance_points[:, 0].max() - instance_points[:, 0].min(),
        instance_points[:, 1].max() - instance_points[:, 1].min(),
        instance_points[:, 2].max() - instance_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (instance_points[:, 0].max() + instance_points[:, 0].min()) * 0.5
    mid_y = (instance_points[:, 1].max() + instance_points[:, 1].min()) * 0.5
    mid_z = (instance_points[:, 2].max() + instance_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set camera view to a reasonable angle for visualization
    ax.view_init(elev=45, azim=135)
    
    # Save or show the plot for debugging
    plt.savefig(f"instance_{instance_id}_{label_id}.png", dpi=150)
    plt.close()


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

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    print(f'loaded checkpoint {args.ckpt_path}')
    sd = checkpoint['module']
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    model.to('cpu')


    target_classes = ["bed",
    "cabinet",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "bathtub",
    "shower curtain",
    "toilet",
    "sink"]

    NYU40_COLORS = [
    (31, 119, 180),     # cabinet
    (255, 187, 120),    # bed
    (188, 189, 34),     # chair
    (140, 86, 75),      # sofa
    (255, 152, 150),    # table
    (214, 39, 40),      # door
    (197, 176, 213),    # window
    (148, 103, 189),    # bookshelf
    (196, 156, 148),    # picture
    (23, 190, 207),     # counter
    (247, 182, 210),    # desk
    (219, 219, 141),    # curtain
    (255, 127, 14),     # refrigerator
    (158, 218, 229),    # shower curtain
    (44, 160, 44),      # toilet
    (112, 128, 144),    # sink
    (227, 119, 194)     # bathtub
]
    
    # Load mesh and data files
    scene_path = "/data/scannet/scans/scene0000_00/scene0000_00_vh_clean.ply"
    aggregation_path = "/data/scannet/scans/scene0000_00/scene0000_00_vh_clean.aggregation.json"
    segmentation_path = "/data/scannet/scans/scene0000_00/scene0000_00_vh_clean.segs.json"

    mesh = trimesh.load(scene_path)
    points = mesh.vertices

    with open(aggregation_path, 'r') as f:
        aggregation_data = json.load(f)
    with open(segmentation_path, 'r') as f:
        seg_data = json.load(f)

    # Map segmentation indices
    seg_to_point_map = seg_data['segIndices']
    object_instances = aggregation_data['segGroups']
    
    # Get text embeddings for target classes
    text_embeddings = generate_text_embeddings(clip_model, target_classes, device, tokenizer_clip, args)

    # Iterate over each instance
    gt_colors, pred_colors = np.ones((points.shape[0], 3)), np.ones((points.shape[0], 3))
    model.to('cuda')
    for i, instance in enumerate(object_instances):
        instance_id = instance['id']
        label_id = instance['label'].lower()
        
        if label_id not in target_classes:
            continue

        # Gather points belonging to this instance
        segments = instance['segments']
        instance_points = []
        for seg in segments:
            seg_indices = np.where(np.isin(seg_to_point_map, seg))[0]
            instance_points.append(points[seg_indices])
            # GT color assignment
            gt_colors[seg_indices] = np.array(NYU40_COLORS[target_classes.index(label_id)]) / 255.0
        
        instance_points = np.concatenate(instance_points, axis=0)
        

        # Classify and assign predicted color
        instance_points_tensor = torch.tensor(instance_points, dtype=torch.float32).unsqueeze(0).to(device)
        pred_label = classify_instance(model, text_embeddings, instance_points_tensor, device)

        for seg in segments:
            seg_indices = np.where(np.isin(seg_to_point_map, seg))[0]
            pred_colors[seg_indices] = np.array(NYU40_COLORS[pred_label]) / 255.0

    from matplotlib.patches import Patch

    for title, colors in zip(["Ground Truth", "Predictions"], [gt_colors, pred_colors]):
        # Filter out points with color (1, 1, 1)
        mask = ~(np.all(colors == [1, 1, 1], axis=1))  # Mask to exclude (1, 1, 1) color points
        filtered_points = points[mask]
        filtered_colors = colors[mask]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot only the filtered points and colors
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], c=filtered_colors, s=1)
        ax.set_title(f"{title} Classification")
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        # Manually adjust the limits of the axes for equal scaling
        max_range = np.array([
            filtered_points[:, 0].max() - filtered_points[:, 0].min(), 
            filtered_points[:, 1].max() - filtered_points[:, 1].min(), 
            filtered_points[:, 2].max() - filtered_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (filtered_points[:, 0].max() + filtered_points[:, 0].min()) * 0.5
        mid_y = (filtered_points[:, 1].max() + filtered_points[:, 1].min()) * 0.5
        mid_z = (filtered_points[:, 2].max() + filtered_points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set camera view to top-down
        ax.view_init(elev=90, azim=-90)  # 90 degrees for top view

        # Create custom legend with class names and colors
        legend_elements = [
            Patch(facecolor=np.array(NYU40_COLORS[i]) / 255.0, label=target_classes[i]) 
            for i in range(len(target_classes))
        ]

        # Add legend to the plot
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.05, 0.5), title="Classes")

        # Save the figure
        plt.savefig(str(title) + ".png", dpi=300, bbox_inches="tight")
        plt.close()



# Load and visualize scene
def load_scene(scene_path):
    mesh = trimesh.load(scene_path)
    points = mesh.vertices
    labels = mesh.metadata['_ply_raw']['vertex']['data']['label']
    return points, labels

def classify_instance(model, text_embeddings, instance_points, device):

    with torch.no_grad():
        instance_points = instance_points.cpu().squeeze().numpy()
        # Subsample the point cloud to the specified number of points
        instance_points = subsample(instance_points, 10000)
        

        instance_points = torch.tensor(instance_points).float().cuda()
        num_points = instance_points.shape[0]
        if  num_points < 10000:
            # Randomly sample indices and duplicate points to reach 10,000
            indices = torch.randint(0, num_points, (10000 - num_points,))
            additional_points = instance_points[indices]
            instance_points = torch.cat([instance_points, additional_points], dim=0)

        # Center and normalize the point cloud to the unit sphere
        instance_points = torch.from_numpy(normalize_to_unit_sphere(instance_points.cpu().numpy())).cuda()

        instance_points[:, [1, 2]] = instance_points[:, [2, 1]]

        rgb = torch.ones_like(instance_points).float() * 0.4

        instance_points = torch.cat((instance_points, rgb),dim=-1)
        pc_features = utils.get_model(model).encode_pc(instance_points.unsqueeze(0)).squeeze()
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        # Compute cosine similarities
        similarities = pc_features.float() @ text_embeddings.float().t()

        pred_label = torch.argmax(similarities).item()
    return pred_label

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


if __name__ == '__main__':
    main(sys.argv[1:])
