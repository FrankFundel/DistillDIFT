import json
import os
import torch

from datasets.dataset import SPairDataset, PFWillowDataset, CUBDataset, ConvertedDataset

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def load_dataset(config, image_transform=None, point_transform=None, from_hdf5=None):
    # Whether to load the dataset from hdf5 or from the original images
    if from_hdf5 is None:
        from_hdf5 = 'from_hdf5' in config and config['from_hdf5']
    if from_hdf5:
        return ConvertedDataset(os.path.join(config['path'], 'converted.h5'), image_transform, point_transform)

    if config['name'] == 'PF-WILLOW':
        return PFWillowDataset(config['path'], image_transform, point_transform)
    if config['name'] == 'SPair-71k':
        return SPairDataset(config['path'], image_transform, point_transform)
    if config['name'] == 'CUB-200-2011':
        return CUBDataset(config['path'], image_transform, point_transform)
    
    raise ValueError('Dataset not recognized.')

class PointResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, points, old_size):
        x_scale = self.size[0] / old_size[0]
        y_scale = self.size[1] / old_size[1]
        rescaled_points = points * torch.tensor([y_scale, x_scale])
        return rescaled_points