import os
import yaml
import tqdm
import h5py
import copy
import torch
import imagesize
from PIL import Image
import torch.utils.data as data

from datasets.correspondence import CorrespondenceDataset, SPair, PFWillow, CUB, S2K
from datasets.image import ImageNet, PASCALPart
from utils.correspondence import preprocess_image, flip_points, flip_bbox, rescale_points, rescale_bbox, normalize_features, flatten_features
from utils.distillation import sample_points

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

    if dataset_name == 'SPair-71k':
        return SPair(config, preprocess)
    if dataset_name == 'PF-WILLOW':
        return PFWillow(config, preprocess)
    if dataset_name == 'CUB-200-2011':
        return CUB(config, preprocess)
    if dataset_name == 'ImageNet':
        return ImageNet(config, preprocess)
    if dataset_name == 'PASCALPart':
        return PASCALPart(config, preprocess)
    if dataset_name == 'S2K':
        return S2K(config, preprocess)
    
    raise ValueError('Dataset not recognized.')

def cache_dataset(model, dataset, cache_path, reset_cache, batch_size, num_workers, device, half_precision):
    """
    Cache features from dataset.

    Args:
        model (CacheModel): Model
        dataset (CorrespondenceDataset): Dataset
        cache_path (str): Path to cache file
        reset_cache (bool): Whether to reset cache
        batch_size (int): Batch size
        num_workers (int): Number of workers for dataloader
        device (torch.device): Device
        half_precision (bool): Whether to use half precision
    """

    if os.path.exists(cache_path) and not reset_cache:
        print(f"Cache file {cache_path} already exists.")
        return CacheDataset(dataset, cache_path)

    print(f"Caching features to {cache_path}")

    with h5py.File(cache_path, 'w') as f:
        keys = []
        samples = []
        def process(image_path, category):
            key = os.path.basename(image_path)
            if key not in keys:
                image = Image.open(image_path)
                image = dataset.preprocess.process_image(image)
                samples.append((key, image, category))
                keys.append(key)
        
        for sample in dataset.data:
            process(sample['source_image_path'], sample['source_category'])
            process(sample['target_image_path'], sample['target_category'])
        
        # Create dataloader
        dataloader = data.DataLoader(samples,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        # Create group for each layer if it doesn't exist
        if hasattr(model, 'layers') and model.layers is not None:
            layers = [f.create_group(str(l)) if str(l) not in f else f[str(l)] for l in model.layers]

        # Move model to device
        model.to(device)

        # Get features and save to cache file
        with torch.no_grad():
            for key, image, category in tqdm.tqdm(dataloader):
                image = image.to(device, dtype=torch.bfloat16 if half_precision else torch.float32)

                # Extend last batch if necessary
                if len(key) < batch_size:
                    image = torch.cat([image, image[-1].repeat(batch_size-len(key), 1, 1, 1)])
                    category = list(category) + [category[-1]] * (batch_size-len(key))
                
                # Get features
                features = model(image, category)

                # Save features
                def save_features(g, features):
                    for i, k in enumerate(key): # for each image and key in the batch
                        g.create_dataset(k, data=features[i].type(torch.float16 if half_precision else torch.float32)) # bfloat16 not supported

                if type(features) is list: # (l, b, c, h, w)
                    for l, layer in enumerate(layers):
                        save_features(layer, features[l].cpu())
                else: # (b, c, h, w)
                    save_features(f, features.cpu())
            
    print("Caching complete.")
    return CacheDataset(dataset, cache_path)

class CacheDataset(CorrespondenceDataset):
    """
    Wrapper for CorrespondenceDataset that loads features from cache instead of images.
    """
    
    def __init__(self, dataset, cache_path):
        self.config = dataset.config
        self.data = dataset.data
        self.preprocess = dataset.preprocess
        self.sample_points = getattr(dataset, 'sample_points', None)
        self.file = h5py.File(cache_path, 'r')
        self.cache = self.file
        self.load_images = False

    def set_layer(self, layer):
        self.cache = self.file[str(layer)]

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # Prevent memory leak

        if self.sample_points is not None:
            self.sample_points(sample)

        if self.load_images:
            # Load image
            sample['source_image'] = Image.open(sample['source_image_path'])
            sample['target_image'] = Image.open(sample['target_image_path'])

            # Save image size
            sample['source_size'] = sample['source_image'].size
            sample['target_size'] = sample['target_image'].size
        else:
            # Get image size quickly
            sample['source_size'] = imagesize.get(sample['source_image_path'])
            sample['target_size'] = imagesize.get(sample['target_image_path'])

        # Load features from cache
        source_key = os.path.basename(sample['source_image_path'])
        target_key = os.path.basename(sample['target_image_path'])
        sample['source_features'] = torch.tensor(self.cache[source_key][()])
        sample['target_features'] = torch.tensor(self.cache[target_key][()])
        
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

    def __init__(self, image_size, preprocess_image=True, image_range=[-1, 1], rescale_data=True, normalize_image=False):
        self.image_size = image_size
        self.preprocess_image = preprocess_image
        self.image_range = image_range
        self.rescale_data = rescale_data
        self.normalize_image = normalize_image

    def process_image(self, image):
        return preprocess_image(image, self.image_size, range=self.image_range, norm=self.normalize_image)

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
