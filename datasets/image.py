import os
import json
import copy
import torch
import scipy.io
import numpy as np
from PIL import Image
import webdataset as wds
import torch.utils.data as data

class ImageDataset(data.Dataset):
    """
    General dataset class for image datasets.
    """

    def __init__(self, config, preprocess=None):
        self.dataset_directory = config['path']
        self.preprocess = preprocess
        self.split = config.get('split', 'test')
        self.image_pair = config.get('image_pair', False)
        self.category_to_id = {}
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
            match_id = np.random.choice(self.category_to_id[sample['category']]) # sample a random image from the same category
            matching_sample = self.data[match_id]
            sample['source_image'], sample['source_size'] = self.load_image(sample['image_path'])
            sample['target_image'], sample['target_size'] = self.load_image(matching_sample['image_path'])
            sample['source_category'] = sample['category']
            sample['target_category'] = matching_sample['category']
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
                    self.data.append({
                        "image_path": sample_path,
                        "category": target
                    })
                    if target not in self.category_to_id:
                        self.category_to_id[target] = []
                    self.category_to_id[target].append(len(self.data) - 1)
            elif self.split == "val":
                syn_id = val_to_syn[entry]
                target = syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.data.append({
                    "image_path": sample_path,
                    "category": target
                })
                if target not in self.category_to_id:
                    self.category_to_id[target] = []
                self.category_to_id[target].append(len(self.data) - 1)

class ImageWebDataset(wds.WebDataset):
    """
    WebDataset class for image datasets.
    """

    def __init__(self, dataset_directory, preprocess=None, split='train', image_pair=False, **kwargs):
        self.dataset_directory = dataset_directory
        self.preprocess = preprocess
        self.split = split
        self.image_pair = image_pair
        self.category_to_id = {}

        shards = [f for f in os.listdir(dataset_directory) if f.endswith('.tar') and split in f]
        super().__init__(shards, **kwargs)

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        size = image.size
        if self.preprocess is not None:
            image = self.preprocess(image)
        return image, size
    
    def __iter__(self):
        for sample in super().__iter__():
            sample = json.loads(sample)
            if self.image_pair:
                match_id = np.random.choice(self.category_to_id[sample['category']])
                matching_sample = self.data[match_id]
                sample['source_image'], sample['source_size'] = self.load_image(sample['image_path'])
                sample['target_image'], sample['target_size'] = self.load_image(matching_sample['image_path'])
                sample['source_category'] = sample['category']
                sample['target_category'] = matching_sample['category']
            else:
                sample['image'], sample['size'] = self.load_image(sample['image_path'])

            yield sample

class PASCALPart(ImageDataset):
    """
    Dataset class for the PASCAL-Part dataset.
    """

    def __init__(self, config, preprocess=None):
        self.single_object = config.get('single_object', False)
        super().__init__(config, preprocess)

    def load_annotations(self, path):
        annotations = scipy.io.loadmat(path)["anno"]
        objects = annotations[0, 0]["objects"]
        objects_list = []

        for obj_idx in range(objects.shape[1]):
            obj = objects[0, obj_idx]

            classname = obj["class"][0]
            mask = obj["mask"]

            parts_list = []
            parts = obj["parts"]
            for part_idx in range(parts.shape[1]):
                part = parts[0, part_idx]
                part_name = part["part_name"][0]
                part_mask = part["mask"]
                parts_list.append({"part_name": part_name, "mask": part_mask})

            objects_list.append({"class": classname, "mask": mask, "parts": parts_list})

        return objects_list

    def load_data(self):
        annotation_directory = os.path.join(self.dataset_directory, "Annotations_Part")
        
        for annotation_filename in os.listdir(annotation_directory):
            annotations = self.load_annotations(os.path.join(annotation_directory, annotation_filename))
            if self.single_object and len(annotations) > 1:
                continue
            image_filename = annotation_filename.replace(".mat", ".jpg")
            image_path = os.path.join(self.dataset_directory, "VOC2010", "JPEGImages", image_filename)

            # get points from part mass centers
            # get bounding box from object mask in (x, y, w, h)
            points = []
            bbox = []
            for obj in annotations:
                for part in obj["parts"]:
                    mask = part["mask"]
                    y, x = np.where(mask)
                    points.append([np.mean(x), np.mean(y)])

                mask = obj["mask"]
                y, x = np.where(mask)
                x1, x2 = min(x), max(x)
                y1, y2 = min(y), max(y)
                bbox.append([x1, y1, x2 - x1, y2 - y1])

                if obj["class"] not in self.category_to_id:
                    self.category_to_id[obj["class"]] = []
                self.category_to_id[obj["class"]].append(len(self.data))

            self.data.append({
                "image_path": image_path,
                "categories": [obj["class"] for obj in annotations], 
                "annotations": annotations,
                "points": torch.tensor(points),
                "bbox": torch.tensor(bbox[0]), # single object
                "category": annotations[0]["class"], # single object
                "mask": annotations[0]["mask"] # single object
            })
    
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        if self.image_pair:
            # make it deterministic
            np.random.seed(23)
            random_category = np.random.choice(sample['categories']) # sample a random category from the image
            match_id = np.random.choice(self.category_to_id[random_category]) # sample a random image from the same category
            matching_sample = self.data[match_id]
            sample['source_image'], sample['source_size'] = self.load_image(sample['image_path'])
            sample['target_image'], sample['target_size'] = self.load_image(matching_sample['image_path'])
            sample['source_category'] = random_category
            sample['target_category'] = random_category
            sample['source_annotations'] = sample['annotations']
            sample['target_annotations'] = matching_sample['annotations']
            sample['source_points'] = sample['points']
            sample['target_points'] = matching_sample['points']
            sample['source_bbox'] = sample['bbox']
            sample['target_bbox'] = matching_sample['bbox']
            sample['source_mask'] = sample['mask']
            sample['target_mask'] = matching_sample['mask']
        else:
            sample['image'], sample['size'] = self.load_image(sample['image_path'])

        return sample