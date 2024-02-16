import yaml

from models.luo import LuoModel
from models.hedlin import HedlinModel
from models.tang import TangModel
from models.zhang import ZhangModel

from models.diffusion import Diffusion
from models.dino import DINO
from models.zoedepth import ZoeDepth
from models.mae import MAE
from models.clip import CLIP
from models.combination import Combination
from models.ensemble import Ensemble
from models.prompt import Prompt
from models.dit import DiT

from models.distilldift import DistillDIFT

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

def load_model(model_name, config):
    """
    Load model from config.

    Args:
        model_name (str): Name of model
        config (dict): Model config
    
    Returns:
        BaseModel: Model
    """

    # Replication models
    if model_name == 'luo':
        return LuoModel(config)
    if model_name == 'hedlin':
        return HedlinModel(config)
    if model_name == 'tang':
        return TangModel(config)
    if model_name == 'zhang':
        return ZhangModel(config)
    
    # Pretrained models
    if model_name.startswith('diff'):
        return Diffusion(config)
    if model_name.startswith('dit'):
        return DiT(config)
    if model_name.startswith('dino'):
        return DINO(config)
    if model_name.startswith('zoedepth'):
        return ZoeDepth(config)
    if model_name.startswith('mae'):
        return MAE(config)
    if model_name.startswith('clip'):
        return CLIP(config)
    
    # Experimenatal models
    if model_name.startswith('combination'):
        return Combination(config,
                           load_model(config['model1'], config['model1_config']),
                           load_model(config['model2'], config['model2_config']))
    if model_name.startswith('ensemble'):
        return Ensemble(config)
    if model_name.startswith('prompt'):
        return Prompt(config)
    
    # Distillation models
    if model_name.startswith('distilldift'):
        return DistillDIFT(config)

    raise ValueError('Model not recognized.')