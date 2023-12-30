import torch
from torch.nn.functional import interpolate

from .base import BaseModel
from utils.correspondence import compute_correspondence

class DINOModel(BaseModel):
    """
    DINO models.

    Args:
        image_size (int): Image size
        version (int): Model version, 1 or 2
        model_size (str): Model size, 's', 'b', 'l', 'g'
        patch_size (int): Patch size
        layers (list): Layers to use
        device (str): Device to run model on
    """
    def __init__(self, image_size, version, model_size, patch_size, layers, device="cuda"):
        super(DINOModel, self).__init__()

        self.image_size = image_size
        self.device = device

        self.version = version
        self.model_size = model_size
        self.patch_size = patch_size
        self.num_patches = image_size[0] // patch_size
        self.layers = layers

        if version == 1:
            repo = 'facebookresearch/dino:main'
            model = 'dino_vit' + model_size + str(patch_size)
        elif version == 2:
            repo = 'facebookresearch/dinov' + str(version)
            model = 'dinov2_vit' + model_size + str(patch_size)
        self.extractor = torch.hub.load(repo, model).to(device)
        self.extractor.eval()

    def get_features(self, image):
        image = (image + 1) / 2 # image is bezween -1 and 1, make image between 0 and 1
        image = interpolate(image, size=self.image_size, mode="bilinear")
        features = self.extractor.get_intermediate_layers( 
            image, 1, return_class_token=False
        )[0].permute(0, 2, 1).reshape(image.shape[0], -1, self.num_patches, self.num_patches)
        return features
    
    def compute_correspondence(self, sample):
        source_features = sample['source_image']
        target_features = sample['target_image']
        source_points = sample['source_points']
        
        predicted_points = compute_correspondence(source_features, target_features, source_points, self.image_size)
        return predicted_points.cpu()

    def __call__(self, sample):
        source_images = sample['source_image']
        target_images = sample['target_image']
        source_points = sample['source_points']
        
        images = torch.cat([source_images, target_images]) #.type(torch.float16)
        
        images = (images + 1) / 2 # image is bezween -1 and 1, make image between 0 and 1
        if self.version == 1:
            features = self.extractor.get_intermediate_layers(images, 1)[0][:, :-1] # remove class token
        elif self.version == 2:
            features = self.extractor.get_intermediate_layers(images, 1, return_class_token=False)[0]
        features = features.permute(0, 2, 1).reshape(images.shape[0], -1, self.num_patches, self.num_patches)
        source_features = features[:len(source_images)]
        target_features = features[len(source_images):]
        
        predicted_points = []
        for i in range(source_features.shape[0]):
            predicted_points.append(compute_correspondence(source_features[i].unsqueeze(0),
                                                           target_features[i].unsqueeze(0),
                                                           source_points[i].unsqueeze(0),
                                                           self.image_size).squeeze(0).cpu())
        return predicted_points