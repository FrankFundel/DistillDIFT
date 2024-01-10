import torch

from .base import CacheModel
from utils.correspondence import compute_correspondence
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
        return torch.cat([interpolate(features1, size=image_size, mode='bilinear'),
                          interpolate(features2, size=image_size, mode='bilinear')
                          ], dim=1)
    
    def compute_correspondence(self, batch):
        predicted_points = []
        batch_size = len(batch['source_image'])
        for b in range(batch_size):
            predicted_points.append(compute_correspondence(batch['source_image'][b].unsqueeze(0),
                                                           batch['target_image'][b].unsqueeze(0),
                                                           batch['source_points'][b].unsqueeze(0),
                                                           batch['source_size'][b],
                                                           batch['target_size'][b])
                                                           .squeeze(0).cpu())
        return predicted_points

    def __call__(self, batch):
        assert len(self.layers) == 1
        images = torch.cat([batch['source_image'], batch['target_image']])

        features = self.get_features(images)[0]
        batch['source_image'] = features[:len(batch['source_image'])]
        batch['target_image'] = features[len(batch['target_image']):]
        
        return self.compute_correspondence(batch)