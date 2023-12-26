import os
import torch
import tqdm
import argparse

import torch.utils.data as data

from utils.dataset import read_dataset_config, load_dataset
from utils.model import read_model_config, load_model
from utils.correspondence import preprocess_image, preprocess_points, preprocess_bbox, compute_pck_img, compute_pck_bbox

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
            pck_img += compute_pck_img(predicted_points[i], target_points[i], image_size, pck_threshold)
            pck_bbox += compute_pck_bbox(predicted_points[i], target_points[i], target_bbox[i], pck_threshold)
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

def cache(model, dataset, cache_path, num_workers):
    print(f"Caching features to {cache_path}")
    with h5py.File(cache_path, 'a') as f:
        keys = list(f.keys()) # Get keys already in cache
        samples = []
        for sample in dataset.data:
            source_key = os.path.basename(sample['source_image_path'])
            target_key = os.path.basename(sample['target_image_path'])
            if source_key not in keys:
                image = Image.open(sample['source_image_path'])
                image = preprocess_image(image, model.image_size)
                samples.append((source_key, image, sample['category']))
                keys.append(source_key)
            if target_key not in keys:
                image = Image.open(sample['target_image_path'])
                image = preprocess_image(image, model.image_size)
                samples.append((target_key, image, sample['category']))
                keys.append(target_key)
        
        batch_size = model.batch_size
        dataloader = data.DataLoader(samples,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        with torch.no_grad():
            for keys, images, categories in tqdm.tqdm(dataloader):
                # extend last batch if necessary
                if len(keys) < batch_size:
                    images = torch.cat([images, images[-1].repeat(batch_size-len(keys), 1, 1, 1)])
                    categories = list(categories) + [categories[-1]] * (batch_size-len(keys))
                images = images.to(device)
                features = model.get_features(images, categories).cpu()
                for i, key in enumerate(keys):
                    f.create_dataset(key, data=features[i])
            
    # Set cache in dataset
    dataset.load_cache(cache_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='luo', help='Name of model to evaluate')
    parser.add_argument('--dataset_config', type=str, default='dataset_config.yaml', help='Path to dataset config file')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='Path to model config file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run model on')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pck_threshold', type=float, default=0.1, help='PCK threshold')
    parser.add_argument('--use_cache', action=argparse.BooleanOptionalAction, default=False, help='Precalculate features and use them for faster evaluation')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory to store cached features')
    parser.add_argument('--num_samples', type=int, default=None, help='Maximum number of samples to evaluate')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_config = args.dataset_config
    model_config = args.model_config
    device_type = args.device
    num_samples = args.num_samples
    num_workers = args.num_workers
    pck_threshold = args.pck_threshold
    use_cache = args.use_cache
    cache_dir = args.cache_dir

    # Load model config
    model_config = read_model_config(model_config)[model_name]

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)
    drop_last_batch = model_config.get('drop_last_batch', False)
    grad_enabled = model_config.get('grad_enabled', False)

    # Load model
    model = load_model(model_name, model_config, device_type)

    device = torch.device(device_type)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    model.to(device)

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)
    
    # Print seperator
    print(f"\n{'='*30} Evaluate {model_name} {'='*40}\n")

    # Evaluate
    for dataset_name in dataset_config:
        print(f"Dataset: {dataset_name}")
        config = dataset_config[dataset_name]
        preprocess = Preprocessor(image_size, preprocess_image = not use_cache)
        dataset = load_dataset(dataset_name, config, preprocess)

        # Limit number of samples if specified
        min_num_samples = min(filter(None, [num_samples, config.get('num_samples', None), len(dataset)]))
        if config.get('random_sampling', False):
            dataset.data = [dataset.data[i] for i in torch.randperm(len(dataset))]
        dataset.data = dataset.data[:min_num_samples]

        # Cache dataset
        if use_cache:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            cache_path = os.path.join(cache_dir, f"{model_name}_{config['name']}.h5")
            cache(model, dataset, cache_path, num_workers)

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
                                     drop_last=drop_last_batch,
                                     collate_fn=collate_fn)

        # with torch.no_grad():
        with torch.set_grad_enabled(grad_enabled):
            pck_img, pck_bbox = evaluate(model, dataloader, image_size, pck_threshold, use_cache)

        print(f"PCK_img: {pck_img * 100:.2f}, PCK_bbox: {pck_bbox * 100:.2f}\n")
