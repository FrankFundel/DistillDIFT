import torch
import tqdm
import argparse

import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode

from utils.correspondence import compute_pck
from utils.dataset import read_config, load_dataset, PointResize

from models.luo import LuoModel
from models.hedlin import HedlinModel

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

        # load images on device
        source_images, target_images = source_images.to(device), target_images.to(device)
        
        source_size = source_images.shape
        target_size = target_images.shape

        # swap from (x, y) to (y, x)
        #source_points = torch.flip(source_points, [2])
        #target_points = torch.flip(target_points, [2])

        # run through model
        predicted_points = model(source_images, target_images, source_points)

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
    parser.add_argument('--dataset_config', type=str, default='dataset_config.json')
    parser.add_argument('--device', type=str, default='cuda')
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
        model = LuoModel(batch_size, device)
    elif model == 'hedlin':
        model = HedlinModel()

    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    model.to(device)

    # Set transforms
    image_transform = Compose([
        Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        ToTensor()
    ])
    point_transform = Compose([
        PointResize((512, 512)),
        ToTensor()
    ])

    # Load dataset config
    dataset_config = read_config(dataset_config)
    
    # Evaluate
    for config in dataset_config:
        print(f"Evaluating dataset: {config['name']}")
        dataset = load_dataset(config, image_transform, point_transform)

        def collate_fn(batch):
            source_images = torch.stack([sample['source_image'] for sample in batch])
            target_images = torch.stack([sample['target_image'] for sample in batch])
            source_points = torch.tensor([sample['source_points'] for sample in batch])
            target_points = torch.tensor([sample['target_points'] for sample in batch])
            return {
                'source_image': source_images,
                'target_image': target_images,
                'source_points': source_points,
                'target_points': target_points
            }

        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        load_size = 512
        with torch.inference_mode():
            pck_img, pck_bbox = evaluate(model, dataloader, load_size, pck_threshold)
        print(f"Dataset: {config['name']}, pck_img: {pck_img}, pck_bbox: {pck_bbox}")
