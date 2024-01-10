import yaml

from models.luo import LuoModel
from models.hedlin import HedlinModel
from models.tang import TangModel
from models.zhang import ZhangModel

from models.diffusion import Diffusion
from models.dino import DINO
from models.zoedepth import ZoeDepth
from models.gan import StyleGAN
from models.mae import MAE
from models.clip import CLIP
from models.combination import Combination

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
    
    if model_name.startswith('diff'):
        return Diffusion(config['model'], config['layers'], config['step'], device_type)
    if model_name.startswith('dino'):
        return DINO(config['version'], config['model_size'], config['patch_size'], config['layers'], device_type)
    if model_name == 'zoedepth':
        return ZoeDepth(config['version'], config['layers'], device_type)
    if model_name.startswith('gan'):
        return StyleGAN(config['model_path'], config['layers'], device_type)
    if model_name == 'mae':
        return MAE(config['model_path'], config['arch'], config['patch_size'], config['layers'], device_type)
    if model_name == 'clip':
        return CLIP(config['layers'], device_type)
    
    if model_name.startswith('combination'):
        return Combination(load_model(config['model1'], config['model1_config'], device_type),
                           load_model(config['model2'], config['model2_config'], device_type),
                           device_type)

    raise ValueError('Model not recognized.')