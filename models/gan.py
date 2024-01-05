import torch
import pickle
from .base import CacheModel
from utils.correspondence import compute_correspondence

####### NOT WORKING YET ########
# https://github.com/justinpinkney/awesome-pretrained-stylegan3
#
# There is no general GAN model available (e.g. GigaGAN or StyleGAN-T),
# StyleGAN-3 is pretrained on faces and not on general images.

class StyleGAN(CacheModel):
    """
    StyleGAN 3 model with GAN inversion (trained encoder or projector).

    Args:
        layers (list): Layers to use
        device (str): Device to run model on
    """
    def __init__(self, layers, device="cuda"):
        super(StyleGAN, self).__init__(device)

        base_path = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'
        model_path = 'stylegan3-r-ffhqu-1024x1024.pkl'
        with open('ffhq.pkl', 'rb') as f:
            self.extractor = pickle.load(f)['G_ema'].cuda()
        self.extractor.eval()

    def get_features(self, image, category):
        # z = torch.randn([1, G.z_dim]).cuda()    # latent codes
        # c = None                                # class labels (not used in this example)
        # img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
        pass

    def compute_correspondence(self, batch):
        pass

    def __call__(self, batch):
        pass