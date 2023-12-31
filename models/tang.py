from .base import CacheModel
from utils.correspondence import compute_correspondence
from replicate.tang.models.dift_sd import SDFeaturizer

class TangModel(CacheModel):
    """
    Model from Tang et al. (https://arxiv.org/abs/2306.03881)
    """
    def __init__(self, device="cuda"):
        super(TangModel, self).__init__(device)

        self.dift = SDFeaturizer(device)
        self.ensemble_size = 8

    def get_features(self, image, category):
        assert len(image) == 1

        prompt = f'a photo of a {category[0]}'
        features = self.dift.forward(image, prompt=prompt, ensemble_size=self.ensemble_size)
        return features
    
    def compute_correspondence(self, batch):
        assert len(batch['source_image']) == 1 and len(batch['target_image']) == 1

        predicted_points = compute_correspondence(batch['source_image'],
                                                  batch['target_image'],
                                                  batch['source_points'][0].unsqueeze(0),
                                                  batch['source_size'][0],
                                                  batch['target_size'][0])
        return predicted_points.cpu()
    
    def __call__(self, batch):
        assert len(batch['source_image']) == 1 and len(batch['target_image']) == 1

        batch['source_image'] = self.get_features(batch['source_image'], batch['category'])
        batch['target_image'] = self.get_features(batch['target_image'], batch['category'])
        return self.compute_correspondence(batch)
