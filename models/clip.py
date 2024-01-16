import torch
from .base import CacheModel
from utils.correspondence import compute_correspondence

import open_clip
from torchvision.transforms import Normalize

class CLIP(CacheModel):
    """
    CLIP model (ViT-L-14-336 by OpenAI).

    Args:
        layers (list): Layers to use
        device (str): Device to run model on
    """
    def __init__(self, layers, device="cuda"):
        super(CLIP, self).__init__(device)
        
        self.patch_size = 14
        self.layers = layers
        self.extractor, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
        self.extractor.eval()

        # Set hooks at the specified layers
        layer_counter = 0
        self.features = {}

        # CLIP vision encoder layers 0-11
        for block in self.extractor.visual.transformer.resblocks:
            if layer_counter in self.layers:
                block.register_forward_hook(self.save_fn(layer_counter))
            layer_counter += 1

    def save_fn(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook
    
    def get_features(self, image, category=None):
        self.features = {}
        image = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(image) # important
        _ = self.extractor.encode_image(image)
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size
        return [l[1:, :].permute(1, 2, 0).reshape(b, -1, h, w) if len(l.shape) == 3 else l for l in self.features.values()]
