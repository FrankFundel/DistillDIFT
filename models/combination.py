import torch

from .base import CacheModel
from torch.nn.functional import interpolate

class Combination(CacheModel):
    """
    Diffusion model.
    """
    def __init__(self, model1, model2, device="cuda"):
        super(Combination, self).__init__(device)
        
        self.model1 = model1
        self.model2 = model2

    def get_features(self, image, category):
        features1 = self.model1.get_features(image * 2 - 1, category) # SD
        if len(features1) == 1:
            features1 = features1[0]
        else:
            image_size = max([f.shape[-1] for f in features1])
            features1 = torch.cat([interpolate(f, size=image_size, mode='bilinear') for f in features1], dim=1)
        
        features2 = self.model2.get_features(image, category) # DINO
        if len(features2) == 1:
            features2 = features2[0]
        else:
            image_size = max([f.shape[-1] for f in features2])
            features2 = torch.cat([interpolate(f, size=image_size, mode='bilinear') for f in features2], dim=1)

        # normalize
        features1 = features1 / features1.norm(dim=1, keepdim=True)
        features2 = features2 / features2.norm(dim=1, keepdim=True)

        # interpolate and concatenate
        image_size = max(features1.shape[-1], features2.shape[-1])
        features1 = interpolate(features1, size=image_size, mode='bilinear')
        features2 = interpolate(features2, size=image_size, mode='bilinear')
        #return [features1, features2, torch.cat([features1, features2], dim=1)]
        return torch.cat([features1, features2], dim=1)
