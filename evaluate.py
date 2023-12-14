import torch
import tqdm
import argparse
import numpy as np

import torch.utils.data as data

from utils.dataset import read_config, load_dataset
from utils.correspondence import preprocess_image, preprocess_points, preprocess_bbox, compute_pck

from models.luo import LuoModel
from models.hedlin import HedlinModel
from models.tang import TangModel

def evaluate(model, dataloader, image_size, pck_threshold):
    model.eval()

    pbar = tqdm.tqdm(total=len(dataloader))

    pck_img = 0
    pck_bbox = 0
    keypoints = 0
    
    for batch in tqdm.tqdm(dataloader):
        # get data
        source_images, target_images = batch['source_image'], batch['target_image']
        source_points, target_points = batch['source_points'], batch['target_points']
        target_bbox = batch['target_bbox']

        # load images on device
        source_images, target_images = source_images.to(device), target_images.to(device)

        # run through model
        predicted_points = model(source_images, target_images, source_points)

        # calculate PCK values
        for i in range(len(source_points)):
            pck_img += compute_pck(predicted_points[i], target_points[i], image_size, pck_threshold)
            pck_bbox += compute_pck(predicted_points[i], target_points[i], image_size, pck_threshold, target_bbox[i])
            keypoints += len(source_points[i])

        # update progress bar
        pbar.update(1)
        pbar.set_postfix({'pck_img': pck_img / keypoints, 'pck_bbox': pck_bbox / keypoints})

    pbar.close()
    return pck_img / keypoints, pck_bbox / keypoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='luo', choices=['luo', 'hedlin', 'tang'])
    parser.add_argument('--dataset_config', type=str, default='dataset_config.json')
    parser.add_argument('--image_size', type=tuple, default=(512, 512))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pck_threshold', type=float, default=0.1)

    # Parse arguments
    args = parser.parse_args()
    model_type = args.model
    dataset_config = args.dataset_config
    image_size = args.image_size
    device_type = args.device
    batch_size = args.batch_size
    num_samples = args.num_samples
    num_workers = args.num_workers
    pck_threshold = args.pck_threshold

    # Load model
    if model_type == 'luo':
        image_size = (224, 224)
        model = LuoModel(batch_size, image_size, device_type)
    elif model_type == 'hedlin':
        image_size = (512, 512)
        batch_size = 1
        model = HedlinModel(image_size, device_type)
    elif model_type == 'tang':
        image_size = (768, 768)
        batch_size = 1
        model = TangModel(image_size, device_type)

    device = torch.device(device_type)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    model.to(device)

    # Load dataset config
    dataset_config = read_config(dataset_config)

    # Define preprocessor
    def preprocess(sample):
        source_size = sample['source_image'].size
        target_size = sample['target_image'].size
        sample['source_image'] = preprocess_image(sample['source_image'], image_size)
        sample['target_image'] = preprocess_image(sample['target_image'], image_size)
        sample['source_points'] = preprocess_points(sample['source_points'], source_size, image_size)
        sample['target_points'] = preprocess_points(sample['target_points'], target_size, image_size)
        sample['source_bbox'] = preprocess_bbox(sample['source_bbox'], source_size, image_size)
        sample['target_bbox'] = preprocess_bbox(sample['target_bbox'], target_size, image_size)
        return sample
    
    # Evaluate
    for config in dataset_config:
        print(f"Evaluating dataset: {config['name']}")
        dataset = load_dataset(config, preprocess)

        if num_samples is not None:
            dataset = data.Subset(dataset, np.arange(num_samples))

        def collate_fn(batch):
            return {
                'source_image': torch.stack([sample['source_image'] for sample in batch]),
                'target_image': torch.stack([sample['target_image'] for sample in batch]),
                'source_points': [sample['source_points'] for sample in batch],
                'target_points': [sample['target_points'] for sample in batch],
                'source_bbox': [sample['source_bbox'] for sample in batch],
                'target_bbox': [sample['target_bbox'] for sample in batch]
            }
        
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        #with torch.no_grad():
        with torch.set_grad_enabled(model_type == 'hedlin'):
            pck_img, pck_bbox = evaluate(model, dataloader, image_size, pck_threshold)

        print(f"Dataset: {config['name']}, pck_img: {pck_img}, pck_bbox: {pck_bbox}")
