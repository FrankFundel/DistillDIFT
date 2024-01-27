import torch

from .base import CacheModel
from extractors.diffusion import SDExtractor
from torchvision.transforms import RandomResizedCrop

from torch.nn.functional import interpolate

class Ensemble(CacheModel):
    """
    Ensemble model.
    """
    def __init__(self, model, layers, steps, ensemble_size, random_cropping, device="cuda"):
        super(Ensemble, self).__init__(device)
        
        self.model = model
        self.layers = layers
        self.steps = steps
        self.ensemble_size = ensemble_size
        self.random_cropping = random_cropping

        self.extractor = SDExtractor(device, model)

    def get_features(self, image, category):
        prompt = [f'a photo of a {c}' for c in category]
        if self.ensemble_size > 1 and len(self.steps) == 1:
            features = {}
            for k in range(self.ensemble_size):
                if self.random_cropping:
                    image_preprocessed = RandomResizedCrop(image.shape[-2:], scale=(0.95, 0.95), ratio=(1.0, 1.0))(image)
                else:
                    image_preprocessed = image
                features[k] = self.extractor(image_preprocessed, prompt=prompt, layers=self.layers, steps=self.steps)[self.steps[0]]
                #features[k] = {0: interpolate(features[k][self.layers[0]], (300, 300), mode="bilinear")}
        else:
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=self.steps)

        features = list(zip(*[s.values() for s in features.values()])) # (steps, layers, b, c, H, W) -> (layers, steps, b, c, H, W)
        features = [torch.stack(l).mean(0) for l in features] # (layers, steps, b, c, H, W) -> (layers, b, c, H, W)
        return features
