import os
import yaml
import tqdm
import h5py
import torch
import imagesize
from PIL import Image
import torch.utils.data as data

from datasets.dataset import CorrespondenceDataset, SPairDataset, PFWillowDataset, CUBDataset
from utils.correspondence import preprocess_image, preprocess_points, preprocess_bbox

def read_dataset_config(config_path):
    """
    Read config from JSON file.

    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Config
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def load_dataset(dataset_name, config, preprocess=None):
    """
    Load dataset from config.

    Args:
        dataset_name (str): Name of dataset
        config (dict): Dataset config
        preprocess (callable): Preprocess function

    Returns:
        CorrespondenceDataset: Dataset
    """

    if dataset_name == 'PF-WILLOW':
        return PFWillowDataset(config['path'], preprocess)
    if dataset_name == 'SPair-71k':
        return SPairDataset(config['path'], preprocess)
    if dataset_name == 'CUB-200-2011':
        return CUBDataset(config['path'], preprocess)
    
    raise ValueError('Dataset not recognized.')

def cache_dataset(model, dataset, cache_path, num_workers):
    """
    Cache features from dataset.
    """

    print(f"Caching features to {cache_path}")

    # Filter out keys already in cache and preprocess images
    with h5py.File(cache_path, 'a') as f:
        keys = list(f.keys())
        samples = []
        def process(image_path, category):
            key = os.path.basename(image_path)
            if key not in keys:
                image = Image.open(image_path)
                image = preprocess_image(image, model.image_size)
                samples.append((key, image, category))
                keys.append(key)
        for sample in dataset.data:
            process(sample['source_image_path'], sample['category'])
            process(sample['target_image_path'], sample['category'])
        
        # Create dataloader
        batch_size = model.batch_size
        dataloader = data.DataLoader(samples,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

        # Get features and save to cache file
        with torch.no_grad():
            for keys, images, categories in tqdm.tqdm(dataloader):
                images = images.to(model.device)
                # Extend last batch if necessary
                if len(keys) < batch_size:
                    images = torch.cat([images, images[-1].repeat(batch_size-len(keys), 1, 1, 1)])
                    categories = list(categories) + [categories[-1]] * (batch_size-len(keys))
                features = model.get_features(images, categories).cpu()
                for i, key in enumerate(keys):
                    f.create_dataset(key, data=features[i])
            
    # Wrap dataset in CacheDataset
    return CacheDataset(dataset, cache_path)

class CacheDataset(CorrespondenceDataset):
    """
    Wrapper for CorrespondenceDataset that loads features from cache instead of images.
    """
    
    def __init__(self, dataset, cache_path):
        self.data = dataset.data
        self.preprocess = dataset.preprocess
        self.cache = h5py.File(cache_path, 'r')

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get image size quickly
        sample['source_size'] = imagesize.get(sample['source_image_path'])
        sample['target_size'] = imagesize.get(sample['target_image_path'])

        # Load features from cache
        source_key = os.path.basename(sample['source_image_path'])
        target_key = os.path.basename(sample['target_image_path'])
        sample['source_image'] = torch.tensor(self.cache[source_key][()])
        sample['target_image'] = torch.tensor(self.cache[target_key][()])
        
        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample
    
class Preprocessor:
    def __init__(self, image_size, preprocess_image=True, preprocess_points=True, preprocess_bbox=True):
        self.image_size = image_size
        self.preprocess_image = preprocess_image
        self.preprocess_points = preprocess_points
        self.preprocess_bbox = preprocess_bbox

    def __call__(self, sample):
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