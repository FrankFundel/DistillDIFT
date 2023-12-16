# BaseModel
# LuoModel
# HedlinModel
# DiffusionModel
# OtherModel
# ...

import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def __call__(self, source_images, target_images, source_points):
        pass
