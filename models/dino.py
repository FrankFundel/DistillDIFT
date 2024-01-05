import torch
from .base import CacheModel
from utils.correspondence import compute_correspondence

class DINO(CacheModel):
    """
    DINO models.

    Args:
        version (int): Model version, 1 or 2
        model_size (str): Model size, 's', 'b', 'l', 'g'
        patch_size (int): Patch size
        layers (list): Layers to use
        device (str): Device to run model on
    """
    def __init__(self, version, model_size, patch_size, layers, registers, device="cuda"):
        super(DINO, self).__init__(device)

        self.version = version
        self.model_size = model_size
        self.patch_size = patch_size
        self.registers = registers
        self.layers = layers

        if version == 1:
            repo = 'facebookresearch/dino:main'
            model = 'dino_vit' + model_size + str(patch_size)
        elif version == 2:
            repo = 'facebookresearch/dinov' + str(version)
            model = 'dinov2_vit' + model_size + str(patch_size)
            if registers:
                model += '_reg'
        self.extractor = torch.hub.load(repo, model).to(device)
        self.extractor.eval()

    def get_features(self, image, category=None):
        b = image.shape[0]
        h = image.shape[2] // self.patch_size
        w = image.shape[3] // self.patch_size
        num_layers_from_bottom = len(self.extractor.blocks) - min(self.layers)

        if self.version == 1:
            features = [f[:, 1:] for f in self.extractor.get_intermediate_layers(image, num_layers_from_bottom)] # remove class token
        elif self.version == 2:
            features = self.extractor.get_intermediate_layers(image, num_layers_from_bottom, return_class_token=False)
        
        return [features[l - min(self.layers)].permute(0, 2, 1).reshape(b, -1, h, w) for l in self.layers]
    
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
