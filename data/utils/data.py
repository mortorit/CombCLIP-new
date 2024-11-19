import numpy as np
import torch


def random_rotate_z(pc):
    # random roate around z axis
    theta = np.random.uniform(0, 2*np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    return np.matmul(pc, R)

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def augment_pc(data):
    data = random_point_dropout(data[None, ...])
    data = random_scale_point_cloud(data)
    data = shift_point_cloud(data)
    data = rotate_perturbation_point_cloud(data)
    data = data.squeeze()
    return data


def random_subsample(pc, sample_size):
    """
    Randomly subsample a point cloud.
    :param pc: the point cloud to subsample. This should be a tensor of shape (n, 3) where n is the number of points in the point cloud.
    :param sample_size: the number of points to sample from the point cloud.
    :return: a tensor of shape (sample_size, 3) containing the subsampled point cloud.
    """
    assert sample_size <= pc.shape[1], "Sample size must be less than the number of points in the point cloud"

    indices = torch.randperm(pc.shape[1])[:sample_size]
    return pc[:, indices]

def get_subsampling_method(method="random"):
    # TODO: Implement furthest point sampling and other methods
    assert method in "random", "Only random subsampling is supported at the moment"

    if method == "random":
        return random_subsample


def combine_pcs(pcs, out_size, min_size_per_sample, subsampling_method: "random", attributes):
    """
    Combine point clouds into a single point cloud of size out_size.
    :param pcs: the point clouds to combine. Each point cloud should be a tensor of shape (n, 3) where n is the number of points in the point cloud
    :param out_size: the number of points in the combined point cloud.
    :param min_size_per_sample: the minimum number of points that should be sampled from each point cloud.
    :param subsampling_method: the method to use for subsampling the points from each point cloud.
    :param attributes: the attributes to apply to each point cloud. This should be a list of dictionaries where each dictionary contains the attributes for a point cloud.
    :return: a tensor of shape (out_size, 3) containing the combined point cloud.
    """
    assert out_size // len(pcs) >= min_size_per_sample, "Too many shapes for the given out shape and minimum size per sample"

    assert len(pcs) == len(attributes), "Number of point clouds and attributes must be the same"

    assert len(pcs) <= 3, "Only 3 point clouds are supported at the moment"


    for attr in attributes:
        assert all([key in ["size", "position"] for key in attr.keys()]), "Unrecognized attribute"

    sample_size = out_size // len(pcs)

    subsampling = get_subsampling_method(subsampling_method)

    last_sample_size = out_size - sample_size * len(pcs) + sample_size

    for i, pc in enumerate(pcs):
        pcs[i] = subsampling(pc, sample_size if i < len(pcs) - 1 else last_sample_size)

        # Random rotate the pc around the vertical axis (index 2)
        angle = torch.rand(1) * 2 * np.pi
        rotation_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle), 0],
                                        [-torch.sin(angle), torch.cos(angle), 0],
                                        [0, 0, 1]
                                        ])
        pcs[i][:3] = torch.mm(rotation_matrix, pcs[i][:3])


    # First adjust sizes according to the attributes: size can be small, normal, or large


    for i, attr in enumerate(attributes):
        if attr["size"] == "small":
            pcs[i][:3] *= 0.5
        elif attr["size"] == "large":
            pcs[i][:3] *= 2

    # Then adjust positions according to the attributes: positions are relative to other objects,
    # the position attribute is a tuple of (potision type, other object index)

    random_add_to_offset = torch.rand(1) * 0.2 - 0.1 # Randomly add a small offset to the position

    for i, attr in enumerate(attributes):
        if attr["position"][0] == "next to":
            # The object is next to the object with the given index, compute the max width of the other object
            next_to_object = attr["position"][1]
            max_width = (pcs[next_to_object][0].max() - pcs[next_to_object][0].min() + 0.2 +
                         pcs[attributes[next_to_object]["position"][1]][0].max()) \
                if attributes[next_to_object]["position"][0] == "next to" else pcs[next_to_object][0].max() + 0.1
            # Place the object so that the minimum of the current object is a bit to the right
            # of the max width of the other object
            to_add = max_width - pcs[i][0].min() + random_add_to_offset
            pcs[i][0] += to_add

            # If the next to object is over another object, move the current object up as well
            if attributes[next_to_object]["position"][0] == "over":
                over_object = attributes[next_to_object]["position"][1]
                max_height = (pcs[over_object][2].max() - pcs[over_object][2].min() + 0.2 +
                              pcs[attributes[over_object]["position"][1]][2].max()) \
                    if attributes[over_object]["position"][0] == "over" else pcs[over_object][2].max() + 0.1
                to_add = max_height - pcs[i][2].min() + random_add_to_offset
                pcs[i][2] += to_add


    for i, attr in enumerate(attributes):
        if attr["position"][0] == "over":
            over_object = attr["position"][1]
            # The object is over the object with the given index, compute the max height of the other object
            max_height = (pcs[over_object][2].max() - pcs[over_object][2].min() + 0.2 +
                            pcs[attributes[over_object]["position"][1]][2].max()) \
                    if attributes[over_object]["position"][0] == "over" else pcs[over_object][2].max() + 0.1
            # Place the object so that the minimum of the current object is a bit above
            # the max height of the other object
            to_add = max_height - pcs[i][2].min() + random_add_to_offset
            pcs[i][2] += to_add

            # If the over object is next to another object, move the current object to the right of the other object
            if attributes[over_object]["position"][0] == "next to":
                next_to_object = attributes[over_object]["position"][1]
                max_width = (pcs[next_to_object][0].max() - pcs[next_to_object][0].min() + 0.2 +
                             pcs[attributes[next_to_object]["position"][1]][0].max()) \
                    if attributes[next_to_object]["position"][0] == "next to" else pcs[next_to_object][0].max() + 0.1
                to_add = max_width - pcs[i][0].min() + random_add_to_offset
                pcs[i][0] += to_add


            # If the over object is shorter than the current, move the over object to the right of a random amount
            # between 0 and 15% of the width of the current object, otherwise move the current object to the right of the over object
            if pcs[over_object][0].max() < pcs[i][0].max():
                to_add = torch.rand(1) * 0.8 * (pcs[i][0].max() - pcs[over_object][0].max())
                pcs[over_object][0] += to_add
            else:
                to_add = torch.rand(1) * 0.8 * (pcs[over_object][0].max() - pcs[i][0].max())
                pcs[i][0] += to_add





    # Finally, combine the point clouds
    combined_pc = torch.cat(pcs, dim=1)

    # Normalize the combined point cloud first 3 dimensions
    combined_pc[:3] -= combined_pc[:3].min()
    combined_pc[:3] /= combined_pc[:3].max()

    return combined_pc