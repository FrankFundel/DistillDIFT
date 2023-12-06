import json
import os

from datasets.dataset import SPairDataset, PFWillowDataset, PreprocessedDataset

def read_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def load_dataset(config, from_hdf5=False, transform=None):
    if from_hdf5:
        return PreprocessedDataset(os.path.join(config['path'], 'preprocessed.h5'), transform)

    if config['name'] == 'PF-WILLOW':
        return PFWillowDataset(config['path'], config['csv_path'], transform)
    if config['name'] == 'SPair-71k':
        return SPairDataset(config['path'], transform)
    
    raise ValueError('Dataset not recognized.')