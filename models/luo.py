import torch
from omegaconf import OmegaConf

from .base import BaseModel
from utils.correspondence import compute_correspondence
from replicate.luo.extract_hyperfeatures import load_models

class LuoModel(BaseModel):
    """
    Model from Luo et al. (https://arxiv.org/abs/2305.14334)
    """
    def __init__(self, batch_size, image_size, device="cuda"):
        super(LuoModel, self).__init__()

        self.batch_size = batch_size * 2
        self.image_size = image_size
        
        config_path = "replicate/luo/configs/test.yaml"

        # Adjust config
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        config["batch_size"] = self.batch_size
        config["load_resolution"] = image_size[0]
        OmegaConf.save(config, config_path)
        
        _, self.diffusion_extractor, self.aggregation_network = load_models(config_path, device)
        self.diffusion_extractor.pipe.enable_attention_slicing()
        self.diffusion_extractor.pipe.enable_xformers_memory_efficient_attention()
        
    def get_features(self, images, categories):
        features, _ = self.diffusion_extractor.forward(images.type(torch.float16))
        b, s, l, w, h = features.shape
        diffusion_hyperfeatures = self.aggregation_network(features.float().view((b, -1, w, h)))
        return diffusion_hyperfeatures
    
    def compute_correspondence(self, batch):
        source_features = batch['source_image']
        target_features = batch['target_image']
        source_points = batch['source_points']

        predicted_points = []
        for i in range(source_features.shape[0]):
            predicted_points.append(compute_correspondence(source_features[i].unsqueeze(0),
                                                           target_features[i].unsqueeze(0),
                                                           source_points[i].unsqueeze(0),
                                                           self.image_size).squeeze(0).cpu())
        return predicted_points

    def __call__(self, batch):
        source_images = batch['source_image']
        target_images = batch['target_image']
        source_points = batch['source_points']
        
        images = torch.cat([source_images, target_images]).type(torch.float16)
        
        features, _ = self.diffusion_extractor.forward(images)
        b, s, l, w, h = features.shape
        diffusion_hyperfeatures = self.aggregation_network(features.float().view((b, -1, w, h)))
        source_features = diffusion_hyperfeatures[:b//2]
        target_features = diffusion_hyperfeatures[b//2:]

        predicted_points = []
        for i in range(source_features.shape[0]):
            predicted_points.append(compute_correspondence(source_features[i].unsqueeze(0),
                                                           target_features[i].unsqueeze(0),
                                                           source_points[i].unsqueeze(0),
                                                           self.image_size).squeeze(0).cpu())
        return predicted_points