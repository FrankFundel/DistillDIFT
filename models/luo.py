import torch
from omegaconf import OmegaConf

from .base import BaseModel
from utils.correspondence import get_correspondences
from replicate.luo.extract_hyperfeatures import load_models

class LuoModel(BaseModel):
    def __init__(self, batch_size, device="cuda"):
        super(LuoModel, self).__init__()

        config_path = "replicate/luo/configs/real.yaml"

        # Adjust config
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        config["batch_size"] = batch_size * 2
        OmegaConf.save(config, config_path)
        
        _, self.diffusion_extractor, self.aggregation_network = load_models(config_path, device)
        self.diffusion_extractor.unet.enable_xformers_memory_efficient_attention()
        
    def __call__(self, source_images, target_images, source_points):
        # images must be tensor and normalized between -1 and 1
        images = torch.cat([source_images, target_images]).type(torch.float16)
        images = 2 * images - 1
        
        features, _ = self.diffusion_extractor.forward(images)
        b, s, l, w, h = features.shape
        diffusion_hyperfeatures = self.aggregation_network(features.float().view((b, -1, w, h)))
        source_features = diffusion_hyperfeatures[:b//2]
        target_features = diffusion_hyperfeatures[b//2:]

        predicted_points = get_correspondences(source_features, target_features, source_points, (512, 512))
        return predicted_points