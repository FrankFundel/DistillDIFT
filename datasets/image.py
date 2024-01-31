import os
import json
import copy
import numpy as np
from PIL import Image
import torch.utils.data as data

class ImageDataset(data.Dataset):
    """
    General dataset class for image datasets.
    """

    def __init__(self, dataset_directory, preprocess=None, split='train', image_pair=False):
        self.dataset_directory = dataset_directory
        self.preprocess = preprocess
        self.split = split
        self.image_pair = image_pair
        self.category_to_path = {}
        self.data = []
        self.load_data()

    def load_data(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        size = image.size
        if self.preprocess is not None:
            image = self.preprocess(image)
        return image, size

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        if self.image_pair:
            target_path = np.random.choice(self.category_to_path[sample['category']]) # sample a random image from the same category
            sample['source_image'], sample['source_size'] = self.load_image(sample['image_path'])
            sample['target_image'], sample['target_size'] = self.load_image(target_path)
        else:
            sample['image'], sample['size'] = self.load_image(sample['image_path'])

        return sample

class ImageNet(ImageDataset):
    """
    Dataset class for the ImageNet dataset.
    """

    def load_data(self):
        syn_to_class = {}

        with open(os.path.join(self.dataset_directory, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = int(class_id)
        
        with open(os.path.join(self.dataset_directory, "ILSVRC2012_val_labels.json"), "rb") as f:
            val_to_syn = json.load(f)

        samples_dir = os.path.join(self.dataset_directory, "ILSVRC2012_" + self.split, "data")
        for entry in os.listdir(samples_dir):
            if self.split == "train":
                syn_id = entry
                target = syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    if target not in self.category_to_path:
                        self.category_to_path[target] = []
                    self.category_to_path[target].append(sample_path)
                    self.data.append({
                        "image_path": sample_path,
                        "category": target
                    })
            elif self.split == "val":
                syn_id = val_to_syn[entry]
                target = syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                if target not in self.category_to_path:
                    self.category_to_path[target] = []
                self.category_to_path[target].append(sample_path)
                self.data.append({
                    "image_path": sample_path,
                    "category": target
                })

