import json
import os

from datasets.dataset import SPairDataset, PFWillowDataset, CUBDataset, ConvertedDataset

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def load_dataset(config, transform=None, from_hdf5=None):
    # Whether to load the dataset from hdf5 or from the original images
    if from_hdf5 is None:
        from_hdf5 = 'from_hdf5' in config and config['from_hdf5']
    if from_hdf5:
        return ConvertedDataset(os.path.join(config['path'], 'converted.h5'), transform)

    if config['name'] == 'PF-WILLOW':
        return PFWillowDataset(config['path'], transform)
    if config['name'] == 'SPair-71k':
        return SPairDataset(config['path'], transform)
    if config['name'] == 'CUB-200-2011':
        return CUBDataset(config['path'], transform)
    
    raise ValueError('Dataset not recognized.')