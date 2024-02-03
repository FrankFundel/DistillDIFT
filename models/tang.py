from .base import CacheModel
from replicate.tang.models.dift_sd import SDFeaturizer

class TangModel(CacheModel):
    """
    Model from Tang et al. (https://arxiv.org/abs/2306.03881)

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(TangModel, self).__init__(config)

        self.device = config["device"]

        self.dift = SDFeaturizer(self.device)
        self.ensemble_size = 8

    def get_features(self, image, category):
        assert len(image) == 1

        prompt = f'a photo of a {category[0]}'
        features = self.dift.forward(image, prompt=prompt, ensemble_size=self.ensemble_size)
        return features
    
    def __call__(self, batch):
        assert len(batch['source_image']) == 1 and len(batch['target_image']) == 1

        batch['source_image'] = self.get_features(batch['source_image'], batch['source_category'])
        batch['target_image'] = self.get_features(batch['target_image'], batch['target_category'])
        return self.compute_correspondence(batch)
