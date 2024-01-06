import torch
from .base import CacheModel
from utils.correspondence import compute_correspondence

import sys
sys.path.append('./thirdparty/mae')

import models_mae
from torchvision.transforms import Normalize

class MAE(CacheModel):
    """
    MAE model.

    Args:
        layers (list): Layers to use
        device (str): Device to run model on
    """
    def __init__(self, model_path, arch, patch_size, layers, device="cuda"):
        super(MAE, self).__init__(device)
        
        self.patch_size = patch_size
        self.layers = layers
        
        # Load model
        self.extractor = getattr(models_mae, arch)()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.extractor.load_state_dict(checkpoint['model'], strict=False)

        # Set hooks at the specified layers
        layer_counter = 0
        self.features = {}

        # Encoder blocks 0-23
        for block in self.extractor.blocks:
            if layer_counter in self.layers:
                block.register_forward_hook(self.save_fn(layer_counter))
            layer_counter += 1

        # Decoder layers 24-31
        for layer in self.extractor.decoder_blocks:
            if layer_counter in self.layers:
                layer.register_forward_hook(self.save_fn(layer_counter))
            layer_counter += 1

    def save_fn(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output
        return hook
    
    def get_features(self, image, category=None):
        self.features = {}
        image = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(image) # important
        _ = self.extractor(image, mask_ratio=0.0)
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size
        return [l[:, 1:].permute(0, 2, 1).reshape(b, -1, h, w) if len(l.shape) == 3 else l for l in self.features.values()]

    def compute_correspondence(self, batch):
        predicted_points = []
        batch_size = len(batch['source_image'])
        for b in range(batch_size):
            predicted_points.append(compute_correspondence(batch['source_image'][b].unsqueeze(0),
                                                           batch['target_image'][b].unsqueeze(0),
                                                           batch['source_points'][b].unsqueeze(0),
                                                           batch['source_size'][b],
                                                           batch['target_size'][b])
                                                           .squeeze(0).cpu())
        return predicted_points

    def __call__(self, batch):
        assert len(self.layers) == 1
        images = torch.cat([batch['source_image'], batch['target_image']])

        features = self.get_features(images)[0]
        batch['source_image'] = features[:len(batch['source_image'])]
        batch['target_image'] = features[len(batch['target_image']):]
        
        return self.compute_correspondence(batch)