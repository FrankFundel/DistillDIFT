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

    def __call__(self, source_images, target_images, source_points):
        category = ''
        prompt = f'a photo of a {category}'
        source_features = self.dift.forward(source_images, prompt=prompt, ensemble_size=self.ensemble_size) # [1, c, h, w]
        target_features = self.dift.forward(target_images, prompt=prompt, ensemble_size=self.ensemble_size) # [1, c, h, w]

        predicted_points = []
        for i in range(len(source_points)):
            predicted_points.append(compute_correspondence(source_features[i].unsqueeze(0),
                                                           target_features[i].unsqueeze(0),
                                                           source_points[i].unsqueeze(0),
                                                           self.image_size).squeeze(0).cpu())
        return predicted_points