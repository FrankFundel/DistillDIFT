import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torchvision

import torch.nn.functional as F
from typing import Tuple
from PIL import Image

def process_image(image_pil, res=None, range=(-1, 1)):
    if res:
        image_pil = image_pil.resize(res, Image.BILINEAR)
    image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min # range [r_min, r_max]
    return image[None, ...], image_pil

def load_image_pair(ann, load_size, device, image_path="", output_size=None):
    img1_pil = Image.open(f"{image_path}/{ann['source_path']}").convert("RGB")
    img2_pil = Image.open(f"{image_path}/{ann['target_path']}").convert("RGB")
    source_size = img1_pil.size
    target_size = img2_pil.size
    ann["source_size"] = source_size
    ann["target_size"] = target_size

    # swap from (x, y) to (y, x)
    if "source_points" in ann:
        source_points, target_points = ann["source_points"], ann["target_points"]
        source_points = np.flip(source_points, 1)
        target_points = np.flip(target_points, 1)
        if output_size is not None:
            source_points = rescale_points(source_points, source_size, output_size)
            target_points = rescale_points(target_points, target_size, output_size)
        else:
            source_points = rescale_points(source_points, source_size, load_size)
            target_points = rescale_points(target_points, target_size, load_size)
    else:
        source_points, target_points = None, None

    img1, img1_pil = process_image(img1_pil, res=load_size)
    img2, img2_pil = process_image(img2_pil, res=load_size)
    img1, img2 = img1.to(device), img2.to(device)
    imgs = torch.cat([img1, img2])
    
    return source_points, target_points, img1_pil, img2_pil, imgs

"""
Helper functions for computing semantic correspondence via nearest neighbors.
"""
def rescale_points(points, old_shape, new_shape):
    # Assumes old_shape and new_shape are in the format (w, h)
    # and points are in (y, x) order
    x_scale = new_shape[0] / old_shape[0]
    y_scale = new_shape[1] / old_shape[1]
    rescaled_points = np.multiply(points, np.array([y_scale, x_scale]))
    return rescaled_points

def flatten_feats(feats):
    # (b, c, w, h) -> (b, w*h, c)
    b, c, w, h = feats.shape
    feats = feats.view((b, c, -1))
    feats = feats.permute((0, 2, 1))
    return feats

def normalize_feats(feats):
    # (b, w*h, c)
    feats = feats / torch.linalg.norm(feats, dim=-1).unsqueeze(-1)
    return feats

def batch_cosine_sim(img1_feats, img2_feats, flatten=True, normalize=True, low_memory=False):
    if flatten:
        img1_feats = flatten_feats(img1_feats)
        img2_feats = flatten_feats(img2_feats)
    if normalize:
        img1_feats = normalize_feats(img1_feats)
        img2_feats = normalize_feats(img2_feats)
    if low_memory:
        sims = []
        for img1_feat in img1_feats[0]:
            img1_sims = img1_feat @ img2_feats[0].T
            sims.append(img1_sims)
        sims = torch.stack(sims)[None, ...]
    else:
        sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))
    return sims

def find_nn_correspondences(sims):
    """
    Assumes sims is shape (b, w*h, w*h). Returns points1 (w*hx2) which indexes the image1 in column-major order
    and points2 which indexes corresponding points in image2.
    """
    w = h = int(math.sqrt(sims.shape[-1]))
    b = sims.shape[0]
    points1 = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h)), dim=-1)
    points1 = points1.expand((b, w, h, 2))
    # Convert from row-major to column-major order
    points1 = points1.reshape((b, -1, 2))
    
    # Note x = col, y = row
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % h
    points2_y = points2 // h
    points2 = torch.stack([points2_y, points2_x], dim=-1)
    
    points1 = points1.to(torch.float32)
    points2 = points2.to(torch.float32)

    return points1, points2

def find_nn_source_correspondences(img1_feats, img2_feats, source_points, load_size):
    """
    Precompute nearest neighbor of source_points in img1 to target_points in img2.
    """
    img1_feats = torch.nn.functional.interpolate(img1_feats, load_size, mode="bilinear")
    img2_feats = torch.nn.functional.interpolate(img2_feats, load_size, mode="bilinear")

    source_idx = torch.from_numpy(points_to_idxs(source_points, load_size)).long()
    # Select source_points in the flattened (w, h) dimension as source_idx
    img1_feats = flatten_feats(img1_feats)
    img2_feats = flatten_feats(img2_feats)
    img1_feats = img1_feats[:, source_idx, :]
    img1_feats = normalize_feats(img1_feats)
    img2_feats = normalize_feats(img2_feats)
    sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))

    # Find nn_correspondences but with points1 = source_points
    num_pixels = int(math.sqrt(sims.shape[-1]))
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % num_pixels
    points2_y = points2 // num_pixels
    points2 = torch.stack([points2_y, points2_x], dim=-1)

    points1 = torch.from_numpy(source_points)
    points2 = points2[0]
    return points1, points2

def points_to_idxs(points, load_size):
    points_y = points[:, 0]
    points_y = np.clip(points_y, 0, load_size[1]-1)
    points_x = points[:, 1]
    points_x = np.clip(points_x, 0, load_size[0]-1)
    idx = load_size[1] * np.round(points_y) + np.round(points_x)
    return idx

def points_to_patches(source_points, num_patches, load_size):
    source_points = np.round(source_points)
    new_H = new_W = num_patches
    # Note that load_size is in (w, h) order and source_points is in (y, x) order
    source_patches_y = (new_H / load_size[1]) * source_points[:, 0]
    source_patches_x = (new_W / load_size[0]) * source_points[:, 1]
    source_patches = np.stack([source_patches_y, source_patches_x], axis=-1)
    # Clip patches for cases where it falls close to the boundary
    source_patches = np.clip(source_patches, 0, num_patches - 1)
    source_patches = np.round(source_patches)
    return source_patches

def compute_pck(predicted_points, target_points, load_size, pck_threshold=0.1, target_bounding_box=None):
    distances = np.linalg.norm(predicted_points - target_points, axis=-1)
    if target_bounding_box is None:
        pck = distances <= pck_threshold * max(load_size)
    else:
        left, top, right, bottom = target_bounding_box
        pck = distances <= pck_threshold * max(right-left, bottom-top)
    return distances, pck, pck.sum() / len(pck)

"""
Helper functions adapted from https://github.com/ShirAmir/dino-vit-features.
"""
def draw_correspondences(points1, points2, image1, image2, image1_label="", image2_label="", title="", radius1=8, radius2=1):
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax1, ax2 = axs[0], axs[1]
    ax1.set_xlabel(image1_label)
    ax2.set_xlabel(image2_label)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('hsv')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    for number, (point1, point2, color) in enumerate(zip(points1, points2, colors)):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axs

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)
