import torch

from .base import BaseModel
from utils.correspondence import compute_correspondence
from extractors.diffusion import SDExtractor

class DiffusionModel(BaseModel):
    """
    Diffusion model.
    """
    def __init__(self, image_size, device="cuda"):
        super(DiffusionModel, self).__init__()
        
        self.image_size = image_size
        self.device = device

        self.extractor = SDExtractor(device=device)

    def get_features(self, image, category):
        pass
    
    def compute_correspondence(self, sample):
        pass

    def __call__(self, sample):
        # Prepare data
        source_images = sample['source_image']
        target_images = sample['target_image']
        source_points = sample['source_points']
        category = sample['category']
        pass