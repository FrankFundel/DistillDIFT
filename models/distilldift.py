import torch
import torch.nn as nn

from .base import CacheModel
from extractors.diffusion import SDExtractor

class LoraLinear(torch.nn.Module):
    def __init__(
        self,
        out_features,
        in_features,
        rank,
    ):
        super().__init__()
        self.rank = rank

        # original weight of the matrix
        self.W = nn.Linear(in_features, out_features, bias=False)
        for p in self.W.parameters():
            p.requires_grad_(False)

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # b should be init wiht 0
        for p in self.B.parameters():
            p.detach().zero_()

    def forward(self, x, y): # y should not be there
        w_out = self.W(x)
        a_out = self.A(x)
        b_out = self.B(a_out)
        return w_out + b_out

class DistillDIFT(CacheModel):
    """
    DistillDIFT model.

    Args:
        config (dict): Model config
    """
    def __init__(self, config):
        super(DistillDIFT, self).__init__(config)
        
        self.model = config["model"]
        self.layers = [1] #config["layers"]
        self.step = config["step"]
        self.weights = config["weights"]
        self.rank = config["rank"]

        self.extractor = SDExtractor(self.model)

        # Freeze all layers
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        self.params_to_optimize = []
        #self.add_lora_to_unet(self.extractor.pipe.unet, ["to_q", "to_k", "to_v", "query", "key", "value"], self.rank)

        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=960, kernel_size=4, stride=3, padding=1, output_padding=0),
            nn.ReLU()
        )
        for p in self.upscaler.parameters():
            self.params_to_optimize.append(p)

        # Load weights
        if self.weights is not None:
            self.load_state_dict(torch.load(self.weights))
    
    def getattr_recursive(self, obj, path):
        parts = path.split('.')
        for part in parts:
            if part.isnumeric():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def add_lora_to_unet(self, unet, target_modules, rank):
        for path, w in unet.state_dict().items():
            if "attn" not in path:
                continue

            if path.split(".")[-2] not in target_modules:
                continue

            attn_module = self.getattr_recursive(unet, ".".join(path.split(".")[:-2]))
            layer_module = self.getattr_recursive(unet, ".".join(path.split(".")[:-1]))

            ll = LoraLinear(
                layer_module.out_features,
                layer_module.in_features,
                rank,
            )

            # W is the original weight matrix
            ll.W.load_state_dict({path.split(".")[-1]: w})

            setattr(
                attn_module,
                path.split(".")[-2],
                ll,
            )

            for p in ll.parameters():
                if p.requires_grad:
                    self.params_to_optimize.append(p)
    
    def forward(self, image, category):
        return self.get_features(image, category)

    def get_features(self, image, category):
        prompt = [f'a photo of a {c}' for c in category]
        features = self.extractor(image, prompt=prompt, layers=self.layers, steps=[self.step])[self.step]
        features = list(features.values())[0] # first layer only
        features = self.upscaler(features)
        return features
