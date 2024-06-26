import torch
from omegaconf import OmegaConf

from .base import CacheModel
from replicate.luo.extract_hyperfeatures import load_models

class LuoModel(CacheModel):
    """
    Model from Luo et al. (https://arxiv.org/abs/2305.14334)

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(LuoModel, self).__init__(config)

        self.batch_size = config["batch_size"]
        self.image_size = config["image_size"]
        self.device = config["device"]
        
        config_path = "replicate/luo/configs/test.yaml"

        # Adjust config
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        config["load_resolution"] = self.image_size[0]
        OmegaConf.save(config, config_path)
        
        _, self.diffusion_extractor, self.aggregation_network = load_models(config_path, self.device)
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
