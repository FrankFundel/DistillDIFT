import torch

from .base import BaseModel
from utils.correspondence import get_correspondences

class HedlinModel(BaseModel):
    def __init__(self):
        super(HedlinModel, self).__init__()
        pass
    
    def __call__(self, source_images, target_images):
        pass