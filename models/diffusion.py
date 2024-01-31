from .base import CacheModel
from extractors.diffusion import SDExtractor

class Diffusion(CacheModel):
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
        if isinstance(self.step, int):
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=[self.step])[self.step]
            return list(features.values())
        else:
            features = self.extractor(image, prompt=prompt, layers=self.layers, steps=self.step)
            return [list(f.values())[0] for f in features.values()]
