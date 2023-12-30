from torch import nn

class BaseModel(nn.Module):
    """
    Base model class.
    """

    def __init__(self, image_size, device="cuda"):
        super(BaseModel, self).__init__()

        self.image_size = image_size
        self.device = device

    def __call__(self, sample):
        raise NotImplementedError

class CacheModel(BaseModel):
    """
    Model that can be cached. Feature extraction and correspondence computation need to be seperable.
    """

    def get_features(self, image, category):
        raise NotImplementedError

    def compute_correspondence(self, sample):
        raise NotImplementedError

    def __call__(self, sample):
        sample['source_image'] = self.get_features(sample['source_image'], sample['category'])
        sample['target_image'] = self.get_features(sample['target_image'], sample['category'])
        return self.compute_correspondence(sample)
