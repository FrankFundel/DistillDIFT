import torch

from .base import BaseModel
from utils.correspondence import get_correspondences
from replicate.luo.extract_hyperfeatures import load_models

class LuoModel(BaseModel):
    def __init__(self, device="cuda"):
        super(LuoModel, self).__init__()
        
        config_path = "replicate/luo/configs/real.yaml"
        _, self.diffusion_extractor, self.aggregation_network = load_models(config_path, device)
        
    def __call__(self, source_images, target_images):
        # images must be tensor and normalized between -1 and 1
        images = torch.cat([source_images, target_images])
        
        features, _ = self.diffusion_extractor.forward(images)
        b, s, l, w, h = features.shape
        diffusion_hyperfeatures = self.aggregation_network(features.float().view((b, -1, w, h)))
        source_features = diffusion_hyperfeatures[:b//2]
        target_features = diffusion_hyperfeatures[b//2:]
    
        predicted_points = get_correspondences(source_features, target_features)
        return predicted_points