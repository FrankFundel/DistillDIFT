import torch
import tqdm
import argparse

from models.luo import LuoModel

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='standard', choices=['luo'], help="The type of model.")
parser.add_argument('device', type=str, default='cuda:0', help='The device to be used.')

# Parse arguments
args = parser.parse_args()
model = args.model
device = args.device

if model == 'luo':
    model = LuoModel(device)
elif model == 'hedlin':
    pass
elif model == 'sd1.4':
    pass

device = torch.device(device)

def evaluate(dataloader):
    for (source_images, target_images, source_points, target_points) in tqdm.tqdm(dataloader):
        # load data on device
        source_images, target_images = source_images.to(device), target_images.to(device)
        source_points, target_points = source_points.to(device), target_points.to(device)
        
        # points must be to be [x, y] tensors
        source_size = source_images.shape
        target_size = target_images.shape

        # swap from (x, y) to (y, x)
        source_points = torch.flip(source_points, 1)
        target_points = torch.flip(target_points, 1)

        # run through model
        predicted_points = model(source_images)
