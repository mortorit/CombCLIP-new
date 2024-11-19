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
import models.uni3d as models

from scannet import PointCloudClassificationDataset

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

    # evaluate model
    if args.evaluate_3d:
        print("=> evaluating...")
        zero_stats, zero_stats_lvis, zero_results_scanobjnn = test_zeroshot_3d(args, model, clip_model, tokenizer_clip)
        print(zero_stats)
        print(zero_stats_lvis)
        print(zero_results_scanobjnn)
        return


def test_zeroshot_3d_core(test_loader, validate_dataset_name, model, clip_model, tokenizer, args=None, test_data=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[validate_dataset_name]

    labels = sorted(labels)

    with torch.no_grad():
        print('=> encoding captions')
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            print(texts)
            texts = tokenizer(texts).to(device=args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top3 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        clip_model.to('cpu')
        model.to('cuda')

        for i, (pc, target, target_name, rgb) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            feature = torch.cat((pc, rgb),dim=-1)
            target = target.to(device=args.device, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(feature)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features.float() @ text_features.float().t()

            # measure accuracy and record loss
            (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
            acc1, acc3, acc5 = utils.scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top3_accurate = correct[:3].float().sum(0, keepdim=True).squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top3_accurate[idx].item():
                    per_class_correct_top3[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top3_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top3_accuracy_per_class[name] = per_class_correct_top3[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top3_accuracy_per_class = collections.OrderedDict(top3_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        print(','.join([str(value) for value in top3_accuracy_per_class.values()]))
        print(','.join([str(value) for value in top5_accuracy_per_class.values()]))
    progress.synchronize()
    print(f'0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}

def test_zeroshot_3d(args, model, clip_model, tokenizer):
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    print(checkpoint)
    print(f'loaded checkpoint {args.ckpt_path}')
    sd = checkpoint['module']
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    train_dataset = get_dataset(None, tokenizer, args, 'train')
    train_dataset.return_caption_raw = True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True,
        collate_fn=customized_collate_fn)

    #text_3d_retrieval(args, train_loader, clip_model, model, tokenizer)


    #tokenizer = SimpleTokenizer()

    test_dataset = utils.get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    test_lvis_dataset = utils.get_dataset(None, tokenizer, args, 'val_lvis')
    test_lvis_loader = torch.utils.data.DataLoader(
        test_lvis_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    test_dataset_scanonjnn = utils.get_dataset(None, tokenizer, args, 'val_scanobjnn')
    test_loader_scanobjnn = torch.utils.data.DataLoader(
        test_dataset_scanonjnn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    test_dataset_scannet = PointCloudClassificationDataset("/data/scannet/instances")
    test_loader_scannet = torch.utils.data.DataLoader(
        test_dataset_scannet, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    results_scannet = test_zeroshot_3d_core(test_loader_scannet, "scannet", model, clip_model, tokenizer, args,
                                         'scannet')

    print('scannet done')
    print(results_scannet)

    results_mnet = test_zeroshot_3d_core(test_loader, args.validate_dataset_name, model, clip_model, tokenizer, args,
                                         'modelnet')
    print('modelnet done')
    print(results_mnet)

    results_scanobjnn = test_zeroshot_3d_core(test_loader_scanobjnn, args.validate_dataset_name_scanobjnn, model,
                                              clip_model, tokenizer, args, 'scanobjnn')
    print('scanobj done')

    results_lvis = test_zeroshot_3d_core(test_lvis_loader, args.validate_dataset_name_lvis, model, clip_model,
                                         tokenizer, args, 'lvis')
    print('lvis done')

    return results_mnet, results_lvis, results_scanobjnn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            if meter.count != 0:
                meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def text_3d_retrieval(args, test_loader, clip_model, model, tokenizer):
    model.eval()
    clip_model.eval()

    top1_text_to_pc, top5_text_to_pc = 0, 0
    top1_pc_to_text, top5_pc_to_text = 0, 0

    with torch.no_grad():
        for i, (point_clouds, captions, rgb, captions_raw) in enumerate(test_loader):
            # Move data to device
            point_clouds = point_clouds.to(args.device)
            rgb = rgb.to(args.device)
            point_clouds = torch.cat((point_clouds, rgb), dim=-1)
            captions = captions.to(args.device)

            # Encode captions and point clouds
            text_embeddings = compute_embedding(clip_model, captions, None, args.device)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1,keepdim=True)

            point_cloud_embeddings = utils.get_model(model).encode_pc(point_clouds)
            point_cloud_embeddings = point_cloud_embeddings / point_cloud_embeddings.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity_matrix = torch.matmul(text_embeddings, point_cloud_embeddings.T)

            # Top-k accuracy for text-to-3D
            top1_correct_text, top5_correct_text = compute_top_k_accuracy(similarity_matrix, k=5)

            top1_text_to_pc += top1_correct_text
            top5_text_to_pc += top5_correct_text

            # Top-k accuracy for 3D-to-text (transpose similarity matrix)
            top1_correct_pc, top5_correct_pc = compute_top_k_accuracy(similarity_matrix.T, k=5)

            top1_pc_to_text += top1_correct_pc
            top5_pc_to_text += top5_correct_pc



            # Visualization
            visualize_retrievals(i, point_clouds, captions_raw, similarity_matrix)

    # Compute accuracy over the entire dataset
    total_batches = len(test_loader)
    print(f'Top-1 Accuracy (Text-to-3D): {top1_text_to_pc / total_batches}')
    print(f'Top-5 Accuracy (Text-to-3D): {top5_text_to_pc / total_batches}')
    print(f'Top-1 Accuracy (3D-to-Text): {top1_pc_to_text / total_batches}')
    print(f'Top-5 Accuracy (3D-to-Text): {top5_pc_to_text / total_batches}')


def compute_top_k_accuracy(similarity_matrix, k=5):
    top_k = similarity_matrix.topk(k=k, dim=-1).indices
    top1_correct = (
                top_k[:, 0] == torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)).float().sum()
    top5_correct = (top_k == torch.arange(similarity_matrix.size(0), device=similarity_matrix.device).unsqueeze(
        -1)).float().sum()

    return top1_correct.item(), top5_correct.item()


def visualize_retrievals(batch_idx, point_clouds, captions, similarity_matrix, save_dir="/storage/retrieval_tests"):
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
    import numpy as np

    # Create directory to save the visualizations
    os.makedirs(save_dir, exist_ok=True)

    # Get the index of the closest point cloud for each caption
    top1_indices = similarity_matrix.argmax(dim=-1)

    fig = plt.figure(figsize=(15, 5))

    # 3D plot for the point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    point_cloud = point_clouds[top1_indices[0]].cpu().numpy()
    xyz = point_cloud[:, :3]  # Get x, y, z coordinates
    rgb = point_cloud[:, 3:] # Normalize RGB values to [0, 1]

    # Scatter plot with colors
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1)  # s=1 sets the point size
    ax1.set_title('Closest Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Display caption as text
    ax2 = fig.add_subplot(122)
    ax2.text(0.5, 0.5, captions[0], horizontalalignment='center', verticalalignment='center', wrap=True)
    ax2.set_title('Caption')
    ax2.set_axis_off()

    # Save figure to disk
    save_path = os.path.join(save_dir, f'retrieval_batch_{batch_idx}.png')
    plt.savefig(save_path)
    plt.close()

    print(f'Saved retrieval visualization for batch {batch_idx} to {save_path}')



if __name__ == '__main__':
    main(sys.argv[1:])
