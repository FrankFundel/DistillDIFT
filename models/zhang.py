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
    
    def get_features(self, image, category):
        assert len(image) == 1 and len(category) == 1

        prompt = f'a photo of a {category[0]}'
        features = self.sd_extractor(image, prompt=prompt, layers=self.layers, steps=self.steps)
        b, n, h, w = features[self.steps[0]][self.layers[-1]].shape
        diffusion_features = []
        for t in self.steps:
            for l in self.layers:
                upsampled_feature = interpolate(features[t][l], size=w, mode="bilinear")
                diffusion_features.append(upsampled_feature)
        diffusion_features = torch.cat(diffusion_features, dim=1)
        diffusion_features = diffusion_features / diffusion_features.norm(dim=1, keepdim=True) # L2 normalize

        # DINO features
        image = (image + 1) / 2 # image is bezween -1 and 1, make image between 0 and 1
        image = interpolate(image, size=self.dino_image_size, mode="bilinear")
        dino_features = self.dino_extractor.get_intermediate_layers( 
            image, 1, return_class_token=False
        )[0].permute(0, 2, 1).reshape(b, -1, self.num_patches, self.num_patches)
        dino_features = interpolate(dino_features, size=w, mode="bilinear")
        dino_features = dino_features / dino_features.norm(dim=1, keepdim=True) # L2 normalize

        features = torch.cat([diffusion_features, dino_features], dim=1)
        return features
    
    def compute_correspondence(self, sample):
        source_features = sample['source_image']
        target_features = sample['target_image']
        source_points = sample['source_points']
        
        assert len(source_features) == 1 and len(target_features) == 1 and len(source_points) == 1
        source_points = source_points[0].unsqueeze(0)

        # w/o PCA it takes 44min to compute correspondence for SPair-71k
        # with PCA it takes 1:20min
        layer_dims = {
            'sd_3': 1280,
            'sd_7': 1280,
            'sd_11': 640,
            'dino': 768
        }
        # co-PCA on diffusion dims in source and target features
        for i, layer in enumerate(list(layer_dims.keys())[:-1]):
            d = layer_dims[layer]
            start = 256 * i
            source_features[:, range(start, start+256)] = self.co_pca(source_features[:, range(start, start+d)], n=self.pca_dim)
            target_features[:, range(start, start+256)] = self.co_pca(target_features[:, range(start, start+d)], n=self.pca_dim)

        predicted_points = compute_correspondence(source_features, target_features, source_points, self.image_size)
        return predicted_points.cpu()

    def __call__(self, sample):
        # Prepare data
        source_images = sample['source_image']
        target_images = sample['target_image']
        source_points = sample['source_points']
        category = sample['category']
        assert len(source_images) == 1 and len(target_images) == 1 and len(source_points) == 1
        source_points = source_points[0].unsqueeze(0)
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
        images = interpolate(images, size=self.dino_image_size, mode="bilinear")
        dino_features = self.dino_extractor.get_intermediate_layers( 
            images, 1, return_class_token=False
        )[0].permute(0, 2, 1).reshape(b, -1, self.num_patches, self.num_patches)
        dino_features = interpolate(dino_features, size=w, mode="bilinear")
        dino_features = dino_features / dino_features.norm(dim=1, keepdim=True) # L2 normalize

        features = torch.cat([diffusion_features, dino_features], dim=1)
        source_features = features[:self.batch_size]
        target_features = features[self.batch_size:]
        
        predicted_points = compute_correspondence(source_features, target_features, source_points, self.image_size)
        return predicted_points.cpu()