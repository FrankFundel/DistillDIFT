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

    """def __call__(self, batch):
        raise NotImplementedError
    """

class CacheModel(BaseModel):
    """
    Model that can be cached. Feature extraction and correspondence computation need to be seperable.
    """

    def get_features(self, image, category):
        raise NotImplementedError

    def compute_correspondence(self, batch):
        if isinstance(batch['source_points'], list):
            predicted_points = []
            batch_size = len(batch['source_image'])
            for b in range(batch_size):
                predicted_points.append(compute_correspondence(batch['source_image'][b].unsqueeze(0),
                                                            batch['target_image'][b].unsqueeze(0),
                                                            batch['source_points'][b].unsqueeze(0),
                                                            batch['source_size'][b],
                                                            batch['target_size'][b])
                                                            .squeeze(0).cpu())
        else: # points are tensors
            predicted_points = compute_correspondence(batch['source_image'],
                                                    batch['target_image'],
                                                    batch['source_points'],
                                                    batch['source_size'],
                                                    batch['target_size']).cpu()
        return predicted_points

"""
    def __call__(self, batch):
        images = torch.cat([batch['source_image'], batch['target_image']])
        categories = batch['source_category'] + batch['target_category']

        features = self.get_features(images, categories)
        if self.layers is not None:
            features = features[0] # only one layer
        batch['source_image'] = features[:len(batch['source_image'])]
        batch['target_image'] = features[len(batch['target_image']):]
        
        return self.compute_correspondence(batch)
"""