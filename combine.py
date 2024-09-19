import numpy as np
import torch

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
        if attr["position"][0] == "next_to":
            # The object is next to the object with the given index, compute the max width of the other object
            next_to_object = attr["position"][1]
            max_width = (pcs[next_to_object][0].max() - pcs[next_to_object][0].min() + 0.2 +
                         pcs[attributes[next_to_object]["position"][1]][0].max()) \
                if attributes[next_to_object]["position"][0] == "next_to" else pcs[next_to_object][0].max() + 0.1
            # Place the object so that the minimum of the current object is a bit to the right
            # of the max width of the other object
            to_add = max_width - pcs[i][0].min() + random_add_to_offset
            pcs[i][0] += to_add


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
            if attributes[over_object]["position"][0] == "next_to":
                next_to_object = attributes[over_object]["position"][1]
                max_width = (pcs[next_to_object][0].max() - pcs[next_to_object][0].min() + 0.2 +
                             pcs[attributes[next_to_object]["position"][1]][0].max()) \
                    if attributes[next_to_object]["position"][0] == "next_to" else pcs[next_to_object][0].max() + 0.1
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

    print(combined_pc.shape, combined_pc[3:])

    return combined_pc