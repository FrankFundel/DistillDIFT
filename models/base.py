from torch import nn

class BaseModel(nn.Module):
    """
    Base model class.
    """

    def __init__(self, device="cuda"):
        super(BaseModel, self).__init__()
        self.device = device

    def __call__(self, batch):
        raise NotImplementedError

class CacheModel(BaseModel):
    """
    Model that can be cached. Feature extraction and correspondence computation need to be seperable.
    """

    def get_features(self, image, category):
        raise NotImplementedError

    def compute_correspondence(self, batch):
        raise NotImplementedError

    def __call__(self, batch):
        raise NotImplementedError
