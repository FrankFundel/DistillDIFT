import torch
import numpy as np
from PIL import Image
from .base import CacheModel
from utils.correspondence import compute_correspondence

import clip
import copy
import torch.nn.functional as F
#from Lafite import dnnlib, legacy

####### NOT WORKING YET ########
# https://github.com/justinpinkney/awesome-pretrained-stylegan3
#
# There is no general GAN model available (e.g. GigaGAN or StyleGAN-T),
# StyleGAN-3 is pretrained on faces and not on general images.

class StyleGAN(CacheModel):
    """
    StyleGAN based model with GAN inversion (projector).

    Args:
        layers (list): Layers to use
        device (str): Device to run model on
    """
    def __init__(self, model_path, layers, device="cuda"):
        super(StyleGAN, self).__init__(device)

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        with dnnlib.util.open_url(model_path) as f:
            network= legacy.load_network_pkl(f)
            self.extractor = network['G_ema'].to(device)
        self.extractor.eval()

    def tensor_to_img(self, tensor):
        img = torch.clamp((tensor + 1.) * 127.5, 0., 255.)
        img_list = img.permute(0, 2, 3, 1)
        img_list = [img for img in img_list]
        return Image.fromarray(torch.cat(img_list, dim=-2).detach().cpu().numpy().astype(np.uint8))
        
    def project(
        self,
        G,
        target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        fts,
        *,
        num_steps                  = 1000,
        w_avg_samples              = 10000,
        initial_learning_rate      = 0.1,
        initial_noise_factor       = 0.05,
        lr_rampdown_length         = 0.25,
        lr_rampup_length           = 0.05,
        noise_ramp_length          = 0.75,
        regularize_noise_weight    = 1e5,
        verbose                    = False,
        device: torch.device
    ):
        assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

        def logprint(*args):
            if verbose:
                print(*args)

        G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

        # Compute w stats.
        logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        # Setup noise inputs.
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        # Features for target image.
        target_images = target.unsqueeze(0).to(device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
        w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            synth_images = G.synthesis(ws, fts=fts, noise_mode='const')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255/2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

            # Save projected W for each optimization step.
            w_out[step] = w_opt.detach()[0]

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        return w_out.repeat([1, G.mapping.num_ws, 1])

    def get_features(self, image, category):
        text_input = f'a photo of a {category[0]}'
        with torch.no_grad():
            tokenized_text = clip.tokenize([text_input]).to(self.device)
            txt_fts = self.clip_model.encode_text(tokenized_text)
            txt_fts = txt_fts / txt_fts.norm(dim=-1, keepdim=True)

        #z = torch.randn([b, self.extractor.z_dim]).to(self.device)
        projected_w_steps = self.project(
            self.extractor,
            target=image[0],
            fts=txt_fts,
            num_steps=1000,
            device=self.device,
            verbose=False
        )
        ws = projected_w_steps[-1].unsqueeze(0)
        img = self.extractor.synthesis(ws, fts=txt_fts, noise_mode='const')
        print(img.shape)

        from matplotlib import pyplot as plt
        plt.imshow(self.tensor_to_img(img))
        plt.savefig('test.png')

    def compute_correspondence(self, batch):
        pass

    def __call__(self, batch):
        self.get_features(batch['source_image'], batch['category'])