import torch

from .base import BaseModel
from utils.correspondence import compute_correspondence
from extractors.diffusion import SDExtractor

class Diffusion(BaseModel):
    """
    Diffusion model.
    """
    def __init__(self, model, layers, step, device="cuda"):
        super(Diffusion, self).__init__(device)
        
        self.model = model
        self.layers = layers
        self.step = step

        self.extractor = SDExtractor(device, model)

    def get_features(self, image, category):
        prompt = [f'a photo of a {c}' for c in category]
        features = self.extractor(image, prompt=prompt, layers=self.layers, steps=[self.step])[self.step]
        return list(features.values())
    
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