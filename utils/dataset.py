import os
import yaml
import tqdm
import h5py
import copy
import torch
import imagesize
from PIL import Image
import torch.utils.data as data

from datasets.dataset import CorrespondenceDataset, SPairDataset, PFWillowDataset, CUBDataset
from utils.correspondence import preprocess_image, flip_points, flip_bbox, rescale_points, rescale_bbox

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

def cache_dataset(model, dataset, cache_path, batch_size, num_workers):
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
                image = dataset.preprocess.process_image(image)
                samples.append((key, image, category))
                keys.append(key)
        for sample in dataset.data:
            process(sample['source_image_path'], sample['category'])
            process(sample['target_image_path'], sample['category'])
        
        # Create dataloader
        dataloader = data.DataLoader(samples,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        # Get features and save to cache file
        with torch.no_grad():
            for keys, image, category in tqdm.tqdm(dataloader):
                image = image.to(model.device)
                # Extend last batch if necessary
                if len(keys) < batch_size:
                    image = torch.cat([image, image[-1].repeat(batch_size-len(keys), 1, 1, 1)])
                    category = list(category) + [category[-1]] * (batch_size-len(keys))

                features = model.get_features(image, category)
                
                # Move features to CPU
                # If features are [l, b, c, h, w], move to CPU separately
                if type(features) is list:
                    features = [[f.cpu() for f in l] for l in features]
                else:
                    features = features.cpu()
                
                for i, key in enumerate(keys):
                    f.create_dataset(key, data=features[i])
            
    print("Caching complete.")
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
        sample = copy.deepcopy(self.data[idx]) # Prevent memory leak

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
    """
    Preprocess dataset samples.

    Args:
        image_size (tuple): (width, height) used for resizing images
        preprocess_image (bool): Whether to preprocess images (resize, normalize, etc.)
        rescale_data (bool): Whether to rescale points and bounding boxes (also sets source_size and target_size to image_size)
    """

    def __init__(self, image_size, preprocess_image=True, image_range=[-1, 1], rescale_data=True):
        self.image_size = image_size
        self.preprocess_image = preprocess_image
        self.image_range = image_range
        self.rescale_data = rescale_data

    def process_image(self, image):
        return preprocess_image(image, self.image_size, range=self.image_range)

    def __call__(self, sample):
        source_size = sample['source_size']
        target_size = sample['target_size']

        # Preprocess images
        if self.preprocess_image:
            sample['source_image'] = self.process_image(sample['source_image'])
            sample['target_image'] = self.process_image(sample['target_image'])

        # Rescale points and bounding boxes
        if self.rescale_data:
            sample['source_points'] = rescale_points(sample['source_points'], source_size, self.image_size)
            sample['target_points'] = rescale_points(sample['target_points'], target_size, self.image_size)
            sample['source_bbox'] = rescale_bbox(sample['source_bbox'], source_size, self.image_size)
            sample['target_bbox'] = rescale_bbox(sample['target_bbox'], target_size, self.image_size)
            sample['source_size'] = self.image_size
            sample['target_size'] = self.image_size
        
        # Flip x, y and w, h axis
        sample['source_points'] = flip_points(sample['source_points'])
        sample['target_points'] = flip_points(sample['target_points'])
        sample['source_bbox'] = flip_bbox(sample['source_bbox'])
        sample['target_bbox'] = flip_bbox(sample['target_bbox'])

        return sample
