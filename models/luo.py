import torch
from omegaconf import OmegaConf

from .base import CacheModel
from utils.correspondence import compute_correspondence
from replicate.luo.extract_hyperfeatures import load_models

class LuoModel(CacheModel):
    """
    Model from Luo et al. (https://arxiv.org/abs/2305.14334)
    """
    def __init__(self, batch_size, image_size, device="cuda"):
        super(LuoModel, self).__init__(device)

        self.batch_size = batch_size
        self.image_size = image_size
        
        config_path = "replicate/luo/configs/test.yaml"

        # Adjust config
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        config["load_resolution"] = image_size[0]
        OmegaConf.save(config, config_path)
        
        _, self.diffusion_extractor, self.aggregation_network = load_models(config_path, device)
        self.diffusion_extractor.pipe.enable_attention_slicing()
        self.diffusion_extractor.pipe.enable_xformers_memory_efficient_attention()
        
    def update_batch_size(self, batch_size):
        if self.diffusion_extractor.batch_size != batch_size:
            self.diffusion_extractor.batch_size = batch_size
            self.diffusion_extractor.change_cond("", "cond")
            self.diffusion_extractor.change_cond("", "uncond")

    def get_features(self, image, category=None):
        self.update_batch_size(image.shape[0])
        features, _ = self.diffusion_extractor.forward(image.type(torch.float16))
        b, s, l, w, h = features.shape
        diffusion_hyperfeatures = self.aggregation_network(features.float().view((b, -1, w, h)))
        return diffusion_hyperfeatures
    
    def compute_correspondence(self, batch):
        predicted_points = []
        for i in range(self.batch_size):
            predicted_points.append(compute_correspondence(batch['source_image'][i].unsqueeze(0),
                                                           batch['target_image'][i].unsqueeze(0),
                                                           batch['source_points'][i].unsqueeze(0),
                                                           batch['source_size'][i],
                                                           batch['target_size'][i])
                                                           .squeeze(0).cpu())
        return predicted_points

    def __call__(self, batch):
        images = torch.cat([batch['source_image'], batch['target_image']]).type(torch.float16)
        diffusion_hyperfeatures = self.get_features(images)
        batch['source_image'] = diffusion_hyperfeatures[:self.batch_size]
        batch['target_image'] = diffusion_hyperfeatures[self.batch_size:]
        return self.compute_correspondence(batch)