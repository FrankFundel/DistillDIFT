import torch
import tqdm
import argparse
import json

import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, CenterCrop

from utils.correspondence import compute_pck
from models.luo import LuoModel
from models.hedlin import HedlinModel

from datasets.dataset import PreprocessedDataset, SPairDataset, PFWillowDataset

def evaluate(model, dataloader, load_size, pck_threshold):
    model.eval()

    pbar = tqdm.tqdm(total=len(dataloader))

    total_ck = 0
    total_ck_img = 0
    total_ck_bbox = 0

    for batch in tqdm.tqdm(dataloader):
        # get data
        source_images, target_images = batch['source_image'], batch['target_image']
        source_points, target_points = batch['source_points'], batch['target_points']

        # load data on device
        source_images, target_images = source_images.to(device), target_images.to(device)
        source_points, target_points = source_points.to(device), target_points.to(device)
        
        source_size = source_images.shape
        target_size = target_images.shape

        # swap from (x, y) to (y, x)
        print(source_points.shape)
        # TODO: THIS IS NOT RIGHT
        source_points = source_points.permute(0, 2, 1)
        target_points = target_points.permute(0, 2, 1)

        # run through model
        predicted_points = model(source_images)

        # calculate PCK values
        target_bounding_box = None
        ck_img, pck_img = compute_pck(predicted_points, target_points, load_size, pck_threshold)
        ck_bbox, pck_bbox = compute_pck(predicted_points, target_points, load_size, pck_threshold, target_bounding_box)
        total_ck_img += ck_img
        total_ck_bbox += ck_bbox
        total_ck += len(target_points)

        # update progress bar
        pbar.update(1)
        pbar.set_description(f"pck_img: {pck_img}, pck_bbox: {pck_bbox}")

    pbar.close()
    return total_ck_img / total_ck, total_ck_bbox / total_ck

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='luo', choices=['luo'])
    parser.add_argument('dataset_config', type=str, default='dataset_config.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pck_threshold', type=float, default=0.1)

    # Parse arguments
    args = parser.parse_args()
    model = args.model
    dataset_config = args.dataset_config
    device = args.device
    batch_size = args.batch_size
    num_workers = args.num_workers
    pck_threshold = args.pck_threshold

    # Load model
    if model == 'luo':
        model = LuoModel(device)
    elif model == 'hedlin':
        model = HedlinModel()

    device = torch.device(device)
    model.to(device)

    # Set transforms
    transform = Compose([
        ToTensor(),
        CenterCrop(256)
    ])

    # Load dataset config
    with open(dataset_config) as f:
        dataset_config = json.load(f)
    
    # Evaluate
    for dataset in dataset_config:
        dataset_name = dataset['name']
        dataset_path = dataset['path']
        load_size = dataset['load_size']

        if dataset_name == 'SPair-71K':
            dataset = SPairDataset(dataset_path, transform=transform)
        elif dataset_name == 'PF-WILLOW':
            dataset = PFWillowDataset(*dataset_path, transform=transform)

        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        pck_img, pck_bbox = evaluate(model, dataloader, load_size, pck_threshold)
        print(f"Dataset: {dataset_name}, pck_img: {pck_img}, pck_bbox: {pck_bbox}")
