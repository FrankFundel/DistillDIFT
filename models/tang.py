from .base import CacheModel
from utils.correspondence import compute_correspondence
from replicate.tang.models.dift_sd import SDFeaturizer

class TangModel(CacheModel):
    """
    Model from Tang et al. (https://arxiv.org/abs/2306.03881)
    """
    def __init__(self, image_size, device="cuda"):
        super(TangModel, self).__init__(image_size, device)

        self.dift = SDFeaturizer(device)
        self.ensemble_size = 8

    def get_features(self, image, category):
        assert len(image) == 1

        prompt = f'a photo of a {category[0]}'
        features = self.dift.forward(image, prompt=prompt, ensemble_size=self.ensemble_size)
        return features
    
    def compute_correspondence(self, sample):
        assert len(sample['source_image']) == 1 and len(sample['target_image']) == 1

        predicted_points = compute_correspondence(sample['source_image'],
                                                  sample['target_image'],
                                                  sample['source_points'][0].unsqueeze(0),
                                                  sample['source_size'][0],
                                                  sample['target_size'][0])
        return predicted_points.cpu()
    
    def __call__(self, sample):
        assert len(sample['source_image']) == 1 and len(sample['target_image']) == 1

        sample['source_image'] = self.get_features(sample['source_image'], sample['category'])
        sample['target_image'] = self.get_features(sample['target_image'], sample['category'])
        return self.compute_correspondence(sample)
