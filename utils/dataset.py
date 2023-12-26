import yaml

from datasets.dataset import SPairDataset, PFWillowDataset, CUBDataset

def read_dataset_config(config_path):
    """
    Read config from JSON file.

    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Config
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def load_dataset(dataset_name, config, preprocess=None):
    """
    Load dataset from config.

    Args:
        dataset_name (str): Name of dataset
        config (dict): Dataset config
        preprocess (callable): Preprocess function

    Returns:
        CorrespondenceDataset: Dataset
    """

    if dataset_name == 'PF-WILLOW':
        return PFWillowDataset(config['path'], preprocess)
    if dataset_name == 'SPair-71k':
        return SPairDataset(config['path'], preprocess)
    if dataset_name == 'CUB-200-2011':
        return CUBDataset(config['path'], preprocess)
    
    raise ValueError('Dataset not recognized.')
