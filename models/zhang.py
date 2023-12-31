import torch
from torch.nn.functional import interpolate

from .base import CacheModel
from utils.correspondence import compute_correspondence
from extractors.diffusion import SDExtractor

class ZhangModel(CacheModel):
    """
    Model from Zhang et al. (https://arxiv.org/abs/2305.14334)
    Using own SD and DINO extractors
    """
    def __init__(self, device="cuda"):
        super(ZhangModel, self).__init__(device)

        self.pca_dim = 256
        self.layers = [3, 7, 11]
        self.steps = [100]
        self.sd_extractor = SDExtractor(device, model='runwayml/stable-diffusion-v1-5')

        self.patch_size = 14
        self.num_patches = 60
        self.dino_image_size = 840
        self.dino_layer = 11
        self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
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
        b, _, h, w = features[self.steps[0]][self.layers[-1]].shape
        diffusion_features = []
        for t in self.steps:
            for l in self.layers:
                upsampled_feature = interpolate(features[t][l], size=(h, w), mode="bilinear")
                diffusion_features.append(upsampled_feature)
        diffusion_features = torch.cat(diffusion_features, dim=1)
        diffusion_features = diffusion_features / diffusion_features.norm(dim=1, keepdim=True) # L2 normalize

        # DINO features
        image = (image + 1) / 2 # image is bezween -1 and 1, make image between 0 and 1
        image = interpolate(image, size=self.dino_image_size, mode="bilinear")
        dino_features = self.dino_extractor.get_intermediate_layers( 
            image, 1, return_class_token=False
        )[0].permute(0, 2, 1).reshape(b, -1, self.num_patches, self.num_patches)
        dino_features = interpolate(dino_features, size=(h, w), mode="bilinear")
        dino_features = dino_features / dino_features.norm(dim=1, keepdim=True) # L2 normalize

        features = torch.cat([diffusion_features, dino_features], dim=1)
        return features
    
    def compute_correspondence(self, batch):
        assert len(batch['source_image']) == 1 and len(batch['target_image']) == 1

        predicted_points = compute_correspondence(batch['source_image'],
                                                  batch['target_image'],
                                                  batch['source_points'][0].unsqueeze(0),
                                                  batch['source_size'][0],
                                                  batch['target_size'][0])
        return predicted_points.cpu()

    def __call__(self, batch):
        source_images = batch['source_image']
        target_images = batch['target_image']
        category = batch['category'][0]

        # Prepare data
        assert len(source_images) == 1 and len(target_images) == 1
        images = torch.cat([source_images, target_images])

        # SD features
        prompt = f'a photo of a {category}'
        features = self.sd_extractor(images, prompt=prompt, layers=self.layers, steps=self.steps)
        
        # Aggregate diffusion features
        b, _, h, w = features[self.steps[0]][self.layers[-1]].shape
        diffusion_features = []
        for t in self.steps:
            for l in self.layers:
                reduced_feature = self.co_pca(features[t][l], n=self.pca_dim) # [2, n_components, H, W]
                upsampled_feature = interpolate(reduced_feature, size=(h, w), mode="bilinear")
                diffusion_features.append(upsampled_feature)
        diffusion_features = torch.cat(diffusion_features, dim=1)
        diffusion_features = diffusion_features / diffusion_features.norm(dim=1, keepdim=True) # L2 normalize
        
        # DINO features
        images = (images + 1) / 2 # image is bezween -1 and 1, make image between 0 and 1
        images = interpolate(images, size=self.dino_image_size, mode="bilinear")
        dino_features = self.dino_extractor.get_intermediate_layers( 
            images, 1, return_class_token=False
        )[0].permute(0, 2, 1).reshape(b, -1, self.num_patches, self.num_patches)
        dino_features = interpolate(dino_features, size=(h, w), mode="bilinear")
        dino_features = dino_features / dino_features.norm(dim=1, keepdim=True) # L2 normalize

        features = torch.cat([diffusion_features, dino_features], dim=1) # concatenate normalized diffusion and dino features
        batch['source_image'] = features[:len(batch['source_image'])]
        batch['target_image'] = features[len(batch['target_image']):]
        return self.compute_correspondence(batch)