import os
import json
import copy
from PIL import Image
import torch.utils.data as data

class ImageDataset(data.Dataset):
    """
    General dataset class for image datasets.
    """

    def __init__(self, dataset_directory, preprocess=None, split='train'):
        self.dataset_directory = dataset_directory
        self.preprocess = preprocess
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        # Load image
        sample['image'] = Image.open(sample['image_path'])

        # Save image size
        sample['size'] = sample['source_image'].size
    
        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample

class ImageNet(ImageDataset):
    """
    Dataset class for the ImageNet dataset.
    """

    def load_data(self):
        data = []
        syn_to_class = {}

        with open(os.path.join(self.dataset_directory, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = int(class_id)
        
        with open(os.path.join(self.dataset_directory, "ILSVRC2012_val_labels.json"), "rb") as f:
            val_to_syn = json.load(f)

        samples_dir = os.path.join(self.dataset_directory, "ILSVRC/Data/CLS-LOC", self.split)
        for entry in os.listdir(samples_dir):
            if self.split == "train":
                syn_id = entry
                target = syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    data.append({
                        "image_path": sample_path,
                        "category": target
                    })
            elif self.split == "val":
                syn_id = val_to_syn[entry]
                target = syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                data.append({
                    "image_path": sample_path,
                    "category": target
                })

        return data

