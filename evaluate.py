import torch
import tqdm
import argparse

from utils.utils import compute_pck
from models.luo import LuoModel
from models.hedlin import HedlinModel

from datasets.spair import SPairDataset
from dataset.pfwillow import PFWillowDataset

def evaluate(model, dataloader, load_size, pck_threshold):
    model.eval()

    pbar = tqdm.tqdm(total=len(dataloader))

    total_ck = 0
    total_ck_img = 0
    total_ck_bbox = 0

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

        # calculate PCK values
        target_bounding_box = None
        ck_img, pck_img = compute_pck(predicted_points, target_points, load_size, pck_threshold)
        ck_bbox, pck_bbox = compute_pck(predicted_points, target_points, load_size, pck_threshold, target_bounding_box)
        total_ck += len(target_points)
        total_ck_img += ck_img
        total_ck_bbox += ck_bbox

        # update progress bar
        pbar.update(1)
        pbar.set_description(f"pck_img: {pck_img}, pck_bbox: {pck_bbox}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='standard', choices=['luo'])
    parser.add_argument('device', type=str, default='cuda:0')
    parser.add_argument('--pck_threshold', type=float, default=0.1)

    # Parse arguments
    args = parser.parse_args()
    model = args.model
    device = args.device
    pck_threshold = args.pck_threshold

    # Load model
    if model == 'luo':
        model = LuoModel(device)
    elif model == 'hedlin':
        model = HedlinModel()

    device = torch.device(device)
    model.to(device)

    # Load datasets
    datasets = []
    datasets.append(SPairDataset("data/spair-71k"))
    datasets.append(PFWillowDataset("data/pf-willow"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='standard', choices=['luo'])
    parser.add_argument('device', type=str, default='cuda:0')
    parser.add_argument('--pck_threshold', type=float, default=0.1)

    # Parse arguments
    args = parser.parse_args()
    model = args.model
    device = args.device
    pck_threshold = args.pck_threshold

    # Load model
    if model == 'luo':
        model = LuoModel(device)
    elif model == 'hedlin':
        model = HedlinModel()

    device = torch.device(device)
    model.to(device)

    # Load datasets
    datasets = []
    datasets.append(SPairDataset("data/spair-71k"))
    datasets.append(PFWillowDataset("data/pf-willow"))
