import torch
from torch import nn

from utils.correspondence import compute_correspondence

class BaseModel(nn.Module):
    """
    Base model class.
    """

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, batch):
        raise NotImplementedError


class CacheModel(BaseModel):
    """
    Model that can be cached. Feature extraction and correspondence computation need to be seperable.
    """

    # I don't know why I need to define this again, but it doesn't work otherwise
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for module in self.children():
            module.to(*args, **kwargs)
        return self
    
    def get_features(self, image, category):
        raise NotImplementedError

    def compute_correspondence(self, batch, return_histograms=False):
        if isinstance(batch['source_points'], list):
            predicted_points = []
            batch_size = len(batch['source_features'])
            for b in range(batch_size):
                predicted_points.append(compute_correspondence(batch['source_features'][b],
                                                            batch['target_features'][b],
                                                            batch['source_points'][b],
                                                            batch['source_size'][b],
                                                            batch['target_size'][b],
                                                            return_histograms,
                                                            batch_mode=False))
        else: # points are tensors
            predicted_points = compute_correspondence(batch['source_features'],
                                                    batch['target_features'],
                                                    batch['source_points'],
                                                    batch['source_size'],
                                                    batch['target_size'])
        return predicted_points

    def forward(self, image, category=None):
        return self.get_features(image, category)