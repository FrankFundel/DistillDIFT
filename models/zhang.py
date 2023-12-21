import torch
from torch.nn.functional import interpolate
from torchvision.transforms import Normalize

from .base import BaseModel
from utils.correspondence import compute_correspondence
from extractors.diffusion import SDExtractor, SDHookExtractor

class ZhangModel(BaseModel):
    """
    Model from Zhang et al. (https://arxiv.org/abs/2305.14334)
    Using own SD and DINO extractors
    """
    def __init__(self, batch_size, image_size, device="cuda"):
        super(ZhangModel, self).__init__()

        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device

        self.pca_dim = 256
        self.layers = [3, 7, 11]
        self.steps = [100]
        self.sd_extractor = SDExtractor(self.device, model='runwayml/stable-diffusion-v1-5')

        self.patch_size = 14
        self.num_patches = 60
        self.dino_image_size = 840
        self.dino_layer = 11
        self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
        self.dino_extractor.eval()

    def co_pca(self, features, n=64):
        # [B, C, H, W] -> [B, n, H, W]
        b, c, h, w = features.shape
        features = features.permute(1, 0, 2, 3).reshape(c, -1).T # [B*H*W, C]
        
        U, S, V = torch.pca_lowrank(features, q=n)
        reduced = torch.matmul(features, V[:, :n])
        reduced = reduced.T.view(n, b, h, w).permute(1, 0, 2, 3)
        return reduced

    def __call__(self, sample):
        # Prepare data
        source_images = sample['source_image']
        target_images = sample['target_image']
        source_points = sample['source_points']
        category = sample['category']
        assert len(source_images) == 1 and len(target_images) == 1 and len(source_points) == 1
        images = torch.cat([source_images, target_images]) #.type(torch.float16)

        # SD features
        prompt = f'a photo of a {category[0]}'
        features = self.sd_extractor(images, prompt=prompt, layers=self.layers, steps=self.steps)
        
        # Aggregate diffusion features
        # Simple concatenation results in an unnecessarily high-dimensional feature (2560+1920+960 = 5440).
        # To reduce the high dimension, we compute PCA across the pair of images for each feature layer, and then upsample
        # lower resolution features (i.e., layer 2 and 5) to be the same as the high resolution one (i.e., layer 8) before concatenation.
        b, n, h, w = features[self.steps[0]][self.layers[-1]].shape
        diffusion_features = []
        for t in self.steps:
            for l in self.layers:
                reduced_feature = self.co_pca(features[t][l], n=self.pca_dim) # [B, n_components, H, W]
                upsampled_feature = interpolate(reduced_feature, size=w, mode="bilinear")
                diffusion_features.append(upsampled_feature)
        diffusion_features = torch.cat(diffusion_features, dim=1)
        diffusion_features = diffusion_features / diffusion_features.norm(dim=1, keepdim=True) # L2 normalize

        # DINO features
        images = (images + 1) / 2 # image is bezween -1 and 1, make image between 0 and 1
        images = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(images)
        images = interpolate(images, size=self.dino_image_size, mode="bilinear")
        dino_features = self.dino_extractor.get_intermediate_layers( 
            images, 2, return_class_token=False
        )[0].permute(0, 2, 1).reshape(b, -1, self.num_patches, self.num_patches)
        dino_features = interpolate(dino_features, size=w, mode="bilinear")
        dino_features = dino_features / dino_features.norm(dim=1, keepdim=True) # L2 normalize

        features = torch.cat([diffusion_features, dino_features], dim=1)
        source_features = features[:self.batch_size]
        target_features = features[self.batch_size:]
        
        predicted_points = compute_correspondence(source_features, target_features, source_points, self.image_size)
        return predicted_points.cpu()