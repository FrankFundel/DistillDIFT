import os
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
from models.zhang import ZhangModel

def evaluate(model, dataloader, image_size, pck_threshold, use_cache=False):
    model.eval()

    pbar = tqdm.tqdm(total=len(dataloader))

    pck_img = 0
    pck_bbox = 0
    keypoints = 0
    
    for batch in dataloader:
        # Load images on device
        batch['source_image'] = batch['source_image'].to(device)
        batch['target_image'] = batch['target_image'].to(device)

        # Run through model
        if use_cache:
            predicted_points = model.compute_correspondence(batch)
        else:
            predicted_points = model(batch)

        # Calculate PCK value
        source_points = batch['source_points']
        target_points = batch['target_points']
        target_bbox = batch['target_bbox']
        for i in range(len(source_points)):
            pck_img += compute_pck(predicted_points[i], target_points[i], image_size, pck_threshold)
            pck_bbox += compute_pck(predicted_points[i], target_points[i], image_size, pck_threshold, target_bbox[i])
            keypoints += len(source_points[i])

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'PCK_img': (pck_img / keypoints) * 100, 'PCK_bbox': (pck_bbox / keypoints) * 100})

    pbar.close()
    return pck_img / keypoints, pck_bbox / keypoints

import h5py
from PIL import Image

class Preprocessor:
    def __init__(self, image_size, preprocess_image=True, preprocess_points=True, preprocess_bbox=True):
        self.image_size = image_size
        self.preprocess_image = preprocess_image
        self.preprocess_points = preprocess_points
        self.preprocess_bbox = preprocess_bbox

    def __call__(self, sample):
        #source_size = sample['source_image'].size
        #target_size = sample['target_image'].size
        source_size = sample['source_size']
        target_size = sample['target_size']
        if self.preprocess_image:
            sample['source_image'] = preprocess_image(sample['source_image'], self.image_size)
            sample['target_image'] = preprocess_image(sample['target_image'], self.image_size)
        if self.preprocess_points:
            sample['source_points'] = preprocess_points(sample['source_points'], source_size, self.image_size)
            sample['target_points'] = preprocess_points(sample['target_points'], target_size, self.image_size)
        if self.preprocess_bbox:
            sample['source_bbox'] = preprocess_bbox(sample['source_bbox'], source_size, self.image_size)
            sample['target_bbox'] = preprocess_bbox(sample['target_bbox'], target_size, self.image_size)
        return sample

def cache(model, dataset, cache_path):
    if not os.path.exists(cache_path):
        print(f"Caching features to {cache_path}")
        keys = {}
        for sample in dataset.data:
            source_key = os.path.basename(sample['source_image_path'])
            target_key = os.path.basename(sample['target_image_path'])
            if source_key not in keys:
                image = Image.open(sample['source_image_path'])
                image = preprocess_image(image, model.image_size)
                keys[source_key] = (image, sample['category'])
            if target_key not in keys:
                image = Image.open(sample['target_image_path'])
                image = preprocess_image(image, model.image_size)
                keys[target_key] = (image, sample['category'])
        
        with h5py.File(cache_path, 'w') as f:
            for key in tqdm.tqdm(keys):
                image, category = keys[key]
                features = model.get_features(image.unsqueeze(0), [category]).squeeze(0).cpu()
                f.create_dataset(key, data=features)
                
    # Set cache in dataset
    dataset.load_cache(cache_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='luo', choices=['luo', 'hedlin', 'tang', 'zhang'])
    parser.add_argument('--dataset_config', type=str, default='dataset_config.json')
    parser.add_argument('--image_size', type=tuple, default=(512, 512))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pck_threshold', type=float, default=0.1)
    parser.add_argument('--use_cache', action=argparse.BooleanOptionalAction)

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
    use_cache = args.use_cache

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
    elif model_type == 'zhang':
        image_size = (960, 960)
        batch_size = 1
        model = ZhangModel(1, image_size, device_type)

    device = torch.device(device_type)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    model.to(device)

    # Load dataset config
    dataset_config = read_config(dataset_config)
    
    # Print seperator
    print(f"\n{'='*30} Evaluate {model_type} {'='*40}\n")

    # Evaluate
    for config in dataset_config:
        print(f"Dataset: {config['name']}")
        preprocess = Preprocessor(image_size, preprocess_image = not use_cache)
        dataset = load_dataset(config, preprocess)

        if num_samples is not None:
            dataset = data.Subset(dataset, np.arange(min(num_samples, len(dataset))))

        # Cache dataset
        if use_cache:
            if not os.path.exists('cache'):
                os.mkdir('cache')
            cache_path = f"cache/{model_type}_{config['name']}.h5"
            cache(model, dataset, cache_path)

        def collate_fn(batch):
            return {
                'source_image': torch.stack([sample['source_image'] for sample in batch]),
                'target_image': torch.stack([sample['target_image'] for sample in batch]),
                'source_points': [sample['source_points'] for sample in batch],
                'target_points': [sample['target_points'] for sample in batch],
                'source_bbox': [sample['source_bbox'] for sample in batch],
                'target_bbox': [sample['target_bbox'] for sample in batch],
                'category': [sample['category'] for sample in batch]
            }
        
        dataloader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     drop_last=(model_type == 'luo'),
                                     collate_fn=collate_fn)

        # with torch.no_grad():
        with torch.set_grad_enabled(model_type == 'hedlin'):
            pck_img, pck_bbox = evaluate(model, dataloader, image_size, pck_threshold, use_cache)

        print(f"PCK_img: {pck_img * 100:.2f}, PCK_bbox: {pck_bbox * 100:.2f}\n")
