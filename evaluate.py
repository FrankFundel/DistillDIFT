import os
import torch
import tqdm
import argparse
from matplotlib import pyplot as plt

import torch.utils.data as data

from utils.dataset import read_dataset_config, load_dataset, cache_dataset, Preprocessor
from utils.model import read_model_config, load_model
from utils.correspondence import compute_pck_img, compute_pck_bbox
from utils.visualization import plot_results

def evaluate(model, dataloader, pck_threshold, layers, use_cache=False):
    model.eval()

    pbar = tqdm.tqdm(total=len(dataloader))

    num_layers = len(layers) if layers is not None else 1
    pck_img = torch.zeros(num_layers)
    pck_bbox = torch.zeros(num_layers)
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

        # Calculate PCK values
        batch_size = len(predicted_points)
        target_points = batch['target_points']
        target_bbox = batch['target_bbox']
        target_size = batch['target_size']
        for b in range(batch_size):
            for l in range(len(layers)) if layers is not None else [None]:
                pck_img[l] += compute_pck_img(predicted_points[b][l], target_points[b], target_size[b], pck_threshold)
                pck_bbox[l] += compute_pck_bbox(predicted_points[b][l], target_points[b], target_bbox[b], pck_threshold)
            keypoints += len(target_points[b])

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'PCK_img': (pck_img.max().item() / keypoints) * 100, 'PCK_bbox': (pck_bbox.max().item() / keypoints) * 100})

    pbar.close()
    return pck_img / keypoints, pck_bbox / keypoints


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
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='Plot results')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_config = args.dataset_config
    model_config = args.model_config
    device_type = args.device
    num_workers = args.num_workers
    pck_threshold = args.pck_threshold
    use_cache = args.use_cache
    cache_dir = args.cache_dir
    num_samples = args.num_samples
    plot = args.plot

    # Load model config
    model_config = read_model_config(model_config)[model_name]

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)
    drop_last_batch = model_config.get('drop_last_batch', False)
    grad_enabled = model_config.get('grad_enabled', False)
    rescale_data = model_config.get('rescale_data', False)
    image_range = model_config.get('image_range', (-1, 1))
    layers = model_config.get('layers', None)

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
    print(f"\n{'='*30} Evaluate {model_name} {'='*30}\n")

    # Evaluate
    for dataset_name in dataset_config:
        print(f"Dataset: {dataset_name}")

        # Load dataset parameters
        config = dataset_config[dataset_name]
        dataset_num_samples = config.get('num_samples', None)
        random_sampling = config.get('random_sampling', False)

        preprocess = Preprocessor(image_size, preprocess_image = not use_cache, image_range=image_range, rescale_data=rescale_data)
        dataset = load_dataset(dataset_name, config, preprocess)

        # Limit number of samples if specified
        min_num_samples = min(filter(None, [num_samples, dataset_num_samples, len(dataset)]))
        if random_sampling:
            torch.manual_seed(42)
            dataset.data = [dataset.data[i] for i in torch.randperm(len(dataset))]
        dataset.data = dataset.data[:min_num_samples]

        # Cache dataset
        if use_cache:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            cache_path = os.path.join(cache_dir, f"{model_name}_{dataset_name}.h5")
            dataset = cache_dataset(model, dataset, cache_path, batch_size, num_workers)

        def collate_fn(batch):
            return {
                'source_image': torch.stack([sample['source_image'] for sample in batch]),
                'target_image': torch.stack([sample['target_image'] for sample in batch]),
                'source_points': [sample['source_points'] for sample in batch],
                'target_points': [sample['target_points'] for sample in batch],
                'source_bbox': [sample['source_bbox'] for sample in batch],
                'target_bbox': [sample['target_bbox'] for sample in batch],
                'category': [sample['category'] for sample in batch],
                'source_size': [sample['source_size'] for sample in batch],
                'target_size': [sample['target_size'] for sample in batch],
            }
        
        dataloader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     drop_last=drop_last_batch,
                                     collate_fn=collate_fn)

        with torch.set_grad_enabled(grad_enabled):
            pck_img, pck_bbox = evaluate(model, dataloader, pck_threshold, layers, use_cache)

        if layers is None:
            print(f"PCK_img: {pck_img[0] * 100:.2f}, PCK_bbox: {pck_bbox[0] * 100:.2f}")
        else:
            for l, i in enumerate(layers):
                print(f"Layer {i}: PCK_img: {pck_img[l] * 100:.2f}, PCK_bbox: {pck_bbox[l] * 100:.2f}")

        if plot:
            if not os.path.exists('plots'):
                os.mkdir('plots')
            plot_results(pck_img, pck_bbox, layers, f"plots/{model_name}_{dataset_name}.png")

    print(f"\n{'='*30} Finished {'='*30}\n")