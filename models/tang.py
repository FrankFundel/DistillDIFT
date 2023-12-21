import torch

from .base import BaseModel
from utils.correspondence import compute_correspondence
from replicate.tang.models.dift_sd import SDFeaturizer

class TangModel(BaseModel):
    """
    Model from Tang et al. (https://arxiv.org/abs/2306.03881)
    """
    def __init__(self, image_size, device="cuda"):
        super(TangModel, self).__init__()

        self.dift = SDFeaturizer(device)

        self.image_size = image_size
        self.ensemble_size = 8

    def get_features(self, image, category):
        assert len(image) == 1 and len(category) == 1

        prompt = f'a photo of a {category[0]}'
        features = self.dift.forward(image, prompt=prompt, ensemble_size=self.ensemble_size)
        return features
    
    def compute_correspondence(self, sample):
        source_features = sample['source_image']
        target_features = sample['target_image']
        source_points = sample['source_points']
        
        assert len(source_features) == 1 and len(target_features) == 1 and len(source_points) == 1
        source_points = source_points[0].unsqueeze(0)

        predicted_points = compute_correspondence(source_features, target_features, source_points, self.image_size)
        return predicted_points.cpu()

    def __call__(self, sample):
        # Prepare data
        source_images = sample['source_image']
        target_images = sample['target_image']
        source_points = sample['source_points']
        category = sample['category']

        assert len(source_images) == 1 and len(target_images) == 1 and len(source_points) == 1 and len(category) == 1
        source_points = source_points[0].unsqueeze(0)

        prompt = f'a photo of a {category[0]}'
        source_features = self.dift.forward(source_images, prompt=prompt, ensemble_size=self.ensemble_size) # [1, c, h, w]
        target_features = self.dift.forward(target_images, prompt=prompt, ensemble_size=self.ensemble_size) # [1, c, h, w]

        predicted_points = compute_correspondence(source_features, target_features, source_points, self.image_size)
        return predicted_points.cpu()