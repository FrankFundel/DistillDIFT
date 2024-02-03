import torch

from .base import BaseModel
from replicate.hedlin.utils.optimize_token import load_ldm, optimize_prompt, run_image_with_tokens_cropped, find_max_pixel_value

class HedlinModel(BaseModel):
    """
    Model from Hedlin et al. (https://arxiv.org/abs/2305.15581)

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(HedlinModel, self).__init__()

        self.device = config["device"]
        self.float16 = config["float16"]

        # Default values from Hedlin et al.
        self.layers = [5, 6, 7, 8]
        self.num_opt_iterations = 5
        self.num_steps = 129
        self.lr = 0.0023755632081200314
        self.noise_level = -8
        self.sigma = 27.97853316316864
        self.flip_prob = 0.0
        self.crop_percent = 93.16549294381423
        self.num_iterations = 20

        self.model = load_ldm(self.device, 'CompVis/stable-diffusion-v1-4', float16=self.float16)
        self.model.enable_xformers_memory_efficient_attention()
    
    def __call__(self, batch):
        source_images = batch['source_image']
        target_images = batch['target_image']
        source_points = batch['source_points']

        # Prepare inputs
        assert len(source_images) == 1 and len(target_images) == 1 and len(source_points) == 1
        source_images, target_images, source_points = source_images[0].unsqueeze(0), target_images[0].unsqueeze(0), source_points[0].unsqueeze(0)
        source_points = source_points[:, :, [1, 0]] # flip x and y axis again
        source_points = source_points.permute(0, 2, 1) # (1, 2, N)
        upsample_res = source_images.shape[2]

        # Initialize
        est_keypoints = -1 * torch.ones_like(source_points)

        for j in range(source_points.shape[2]):
            all_maps = []
            for _ in range(self.num_opt_iterations):
                # Optimize the text embeddings for the source point
                context = optimize_prompt(self.model, source_images[0], source_points[0, :, j] / upsample_res,
                                          num_steps=self.num_steps, device=self.device, layers=self.layers, lr = self.lr,
                                          upsample_res=upsample_res, noise_level=self.noise_level, sigma=self.sigma,
                                          flip_prob=self.flip_prob, crop_percent=self.crop_percent)
                
                # Find the attention maps over the optimized text embeddings
                attn_maps, _ = run_image_with_tokens_cropped(self.model, target_images[0], context, index=0, upsample_res=upsample_res,
                                                             noise_level=self.noise_level, layers=self.layers, device=self.device,
                                                             crop_percent=self.crop_percent, num_iterations=self.num_iterations,
                                                             image_mask = None)
                all_maps.append(attn_maps.mean(dim=1, keepdim=True))

            all_maps = torch.stack(all_maps, dim=0)
            all_maps = torch.mean(all_maps, dim=0)
            all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(self.layers), upsample_res * upsample_res))
            all_maps = all_maps.reshape(len(self.layers), upsample_res, upsample_res)

            # Take the argmax to find the corresponding location for the target image
            all_maps = torch.mean(all_maps, dim=0)
            max_val = find_max_pixel_value(all_maps, img_size=upsample_res)
            est_keypoints[0, :, j] = (max_val+0.5)

        est_keypoints = est_keypoints.permute(0, 2, 1) # (1, N, 2)
        est_keypoints = est_keypoints[:, :, [1, 0]] # flip x and y axis again
        return est_keypoints