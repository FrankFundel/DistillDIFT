import yaml

from models.luo import LuoModel
from models.hedlin import HedlinModel
from models.tang import TangModel
from models.zhang import ZhangModel

from models.dino import DINOModel

def read_model_config(config_path):
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

def load_model(model_name, config, device_type):
    """
    Load model from config.

    Args:
        model_name (str): Name of model
        config (dict): Model config
        device_type (str): Device type
    
    Returns:
        BaseModel: Model
    """

    if model_name == 'luo':
        return LuoModel(config['batch_size'], config['image_size'], device_type)
    if model_name == 'hedlin':
        return HedlinModel(device_type)
    if model_name == 'tang':
        return TangModel(device_type)
    if model_name == 'zhang':
        return ZhangModel(device_type)
    
    if model_name.startswith('dino'):
        return DINOModel(config['version'], config['model_size'], config['patch_size'], config['layers'], device_type)

    raise ValueError('Model not recognized.')