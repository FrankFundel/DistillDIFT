import torch
from torch.nn.functional import interpolate

from .base import BaseModel
from utils.correspondence import compute_correspondence
from extractors.diffusion import SDExtractor

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

        self.layers = [2, 5, 8]
        self.steps = [100]
        self.sd_extractor = SDExtractor(self.device, model='runwayml/stable-diffusion-v1-5')

        self.patch_size = 14
        self.num_patches = image_size[0] // self.patch_size
        self.dino_image_size = self.patch_size * self.num_patches
        self.dino_layer = 11
        self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_extractor.eval()

    def co_pca(self, features, n=64):
        # [B, C, H, W] -> [B, n, H, W]
        b, c, h, w = features.shape
        features = features.permute(1, 0, 2, 3).reshape(c, -1).T # [B*H*W, C]
        
        U, S, V = torch.pca_lowrank(features, q=n)
        reduced = torch.matmul(features, V[:, :n])
        reduced = reduced.T.view(n, b, h, w).permute(1, 0, 2, 3)
        return reduced

    def __call__(self, source_images, target_images, source_points):
        images = torch.cat([source_images, target_images]) #.type(torch.float16)

        # SD features
        features = self.sd_extractor(images, prompt='An picture', layers=self.layers, steps=self.steps)
        
        # Aggregate diffusion features
        # Simple concatenation results in an unnecessarily high-dimensional feature (2560+1920+960 = 5440).
        # To reduce the high dimension, we compute PCA across the pair of images for each feature layer, and then upsample
        # lower resolution features (i.e., layer 2 and 5) to be the same as the high resolution one (i.e., layer 8) before concatenation.
        b, n, h, w = features[self.steps[0]][self.layers[-1]].shape
        diffusion_features = []
        for t in self.steps:
            for l in self.layers:
                if l < self.layers[-1]:
                    reduced_feature = self.co_pca(features[t][l], n=64) # [B, n_components, H, W]
                    upsampled_feature = interpolate(reduced_feature, size=w, mode="bilinear")
                    diffusion_features.append(upsampled_feature)
                else:
                    diffusion_features.append(features[t][l])
        diffusion_features = torch.cat(diffusion_features, dim=1)

        # DINO features
        #images = interpolate(images, size=self.dino_image_size, mode="bilinear")
        #dino_features = self.dino_extractor.get_intermediate_layers( 
        #    images, 2, return_class_token=False
        #)[0].permute(0, 2, 1).reshape(b, -1, self.num_patches, self.num_patches)
        #dino_features = interpolate(dino_features, size=w, mode="bilinear")

        features = diffusion_features #torch.cat([diffusion_features, dino_features], dim=1)
        source_features = features[:self.batch_size]
        target_features = features[self.batch_size:]
        
        predicted_points = []
        for i in range(self.batch_size):
            predicted_points.append(compute_correspondence(source_features[i].unsqueeze(0),
                                                           target_features[i].unsqueeze(0),
                                                           source_points[i].unsqueeze(0),
                                                           self.image_size).squeeze(0).cpu())
        return predicted_points