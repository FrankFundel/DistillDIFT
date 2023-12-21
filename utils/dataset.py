import json
import os

from datasets.dataset import SPairDataset, PFWillowDataset, CUBDataset, ConvertedDataset

def read_config(config_path):
    """
    Read config from JSON file.

    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Config
    """
    with open(config_path) as f:
        config = json.load(f)
    return config

def load_dataset(config, preprocess=None, from_hdf5=None):
    """
    Load dataset from config.

    Args:
        config (dict): Dataset config
        preprocess (callable): Preprocess function
        from_hdf5 (bool): Whether to load from HDF5 file

    Returns:
        CorrespondenceDataset: Dataset
    """

    if from_hdf5 is None:
        from_hdf5 = 'from_hdf5' in config and config['from_hdf5']
    if from_hdf5:
        return ConvertedDataset(os.path.join(config['path'], 'converted.h5'), preprocess)

    if config['name'] == 'PF-WILLOW':
        return PFWillowDataset(config['path'], preprocess)
    if config['name'] == 'SPair-71k':
        return SPairDataset(config['path'], preprocess)
    if config['name'] == 'CUB-200-2011':
        return CUBDataset(config['path'], preprocess)
    
    raise ValueError('Dataset not recognized.')
