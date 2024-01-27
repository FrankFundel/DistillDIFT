import torch

from .base import CacheModel
from extractors.diffusion import SDExtractor

class DistillDIFT(CacheModel):
    """
    DistillDIFT model.
    """
    def __init__(self, model, device="cuda"):
        super(DistillDIFT, self).__init__(device)
        self.model = model
        self.extractor = None # TODO: Implement UNet

    def get_features(self, image, category=None):
        return self.extractor(image)
