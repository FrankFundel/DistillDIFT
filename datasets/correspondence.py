import itertools
import torch
import torch.utils.data as data
import numpy as np
import json
import os
import csv
from PIL import Image
import copy

class CorrespondenceDataset(data.Dataset):
    """
    General dataset class for datasets with correspondence points.
    """

    def __init__(self, config, preprocess=None):
        self.config = config
        self.dataset_directory = config['path']
        self.image_pair = config.get('image_pair', False)
        self.preprocess = preprocess
        self.split = config.get('split', 'test')
        self.data = self.load_data()

    def load_data(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        # Load image
        sample['source_image'] = Image.open(sample['source_image_path'])
        sample['target_image'] = Image.open(sample['target_image_path'])

        # Save image size
        sample['source_size'] = sample['source_image'].size
        sample['target_size'] = sample['target_image'].size
    
        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample


class SPair(CorrespondenceDataset):
    """
    Dataset class for the SPair-71k dataset.
    """

    def load_data(self):
        images_dir = os.path.join(self.dataset_directory, 'JPEGImages')
        annotations_dir = os.path.join(self.dataset_directory, 'PairAnnotation', 'test' if self.split == 'test' else 'trn')
        annotations_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]

        data = []
        for annotation_file in annotations_files:
            with open(os.path.join(annotations_dir, annotation_file), 'r') as file:
                annotation = json.load(file)

            category = annotation['category']
            source_image_path = os.path.join(images_dir, category, annotation['src_imname'])
            target_image_path = os.path.join(images_dir, category, annotation['trg_imname'])
            source_points = torch.tensor(annotation['src_kps'], dtype=torch.float16)
            target_points = torch.tensor(annotation['trg_kps'], dtype=torch.float16)
            source_bbox = torch.tensor(annotation['src_bndbox'], dtype=torch.float16)
            target_bbox = torch.tensor(annotation['trg_bndbox'], dtype=torch.float16)

            # Convert from (x, y, x+w, y+h) to (x, y, w, h)
            source_bbox[2:] -= source_bbox[:2]
            target_bbox[2:] -= target_bbox[:2]

            data.append({
                'source_image_path': source_image_path,
                'target_image_path': target_image_path,
                'source_points': source_points,
                'target_points': target_points,
                'source_bbox': source_bbox,
                'target_bbox': target_bbox,
                'source_category': category,
                'target_category': category
            })
        return data


class PFWillow(CorrespondenceDataset):
    """
    Dataset class for the PF-Willow dataset.
    """
    
    def load_data(self):
        data = []
        csv_file = os.path.join(self.dataset_directory, 'test_pairs_pf.csv')
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                source_image_path = os.path.join(self.dataset_directory, row[0].replace('PF-dataset/', ''))
                target_image_path = os.path.join(self.dataset_directory, row[1].replace('PF-dataset/', ''))

                row[2:] = list(map(float, row[2:]))
                source_points = torch.tensor(list(zip(row[2:12], row[12:22])), dtype=torch.float16) # X, Y
                target_points = torch.tensor(list(zip(row[22:32], row[32:])), dtype=torch.float16) # X, Y

                # use min and max to get the bounding box
                source_bbox = torch.tensor([source_points[:, 0].min(), source_points[:, 1].min(),
                                            source_points[:, 0].max(), source_points[:, 1].max()], dtype=torch.float16)
                target_bbox = torch.tensor([target_points[:, 0].min(), target_points[:, 1].min(),
                                            target_points[:, 0].max(), target_points[:, 1].max()], dtype=torch.float16)

                # Convert from (x, y, x+w, y+h) to (x, y, w, h)
                source_bbox[2:] -= source_bbox[:2]
                target_bbox[2:] -= target_bbox[:2]

                # Get category from image path
                category = source_image_path.split('(')[0]

                data.append({
                    'source_image_path': source_image_path,
                    'target_image_path': target_image_path,
                    'source_points': source_points,
                    'target_points': target_points,
                    'source_bbox': source_bbox,
                    'target_bbox': target_bbox,
                    'source_category': category,
                    'target_category': category
                })
        return data


class CUB(CorrespondenceDataset):
    """
    Dataset class for the CUB-200-2011 dataset.
    """

    def load_data(self):
        self.images_dir = os.path.join(self.dataset_directory, 'images')

        with open(os.path.join(self.dataset_directory, "images.txt"), "r") as f:
            images = [line.strip().split() for line in f.readlines()]

        with open(os.path.join(self.dataset_directory, "train_test_split.txt"), "r") as f:
            train_test_split = [line.strip().split() for line in f.readlines()]

        with open(os.path.join(self.dataset_directory, "parts/part_locs.txt"), "r") as f:
            part_locs = {}
            for line in f.readlines():
                img_id, _, x, y, visible = line.strip().split()
                if img_id not in part_locs:
                    part_locs[img_id] = []
                part_locs[img_id].append((x, y, visible == '1'))

        with open(os.path.join(self.dataset_directory, "image_class_labels.txt"), "r") as f:
            image_class_labels = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}

        with open(os.path.join(self.dataset_directory, "bounding_boxes.txt"), "r") as f:
            bounding_boxes = {line.split()[0]: list(map(float, line.strip().split()[1:])) for line in f.readlines()}

        # Filter images based on train/test split and class labels
        filtered_images = []
        for img_id, img_name in images:
            is_training_image = int(train_test_split[int(img_id) - 1][1])
            class_id = image_class_labels[img_id]
            if (self.split == 'train' and is_training_image) or (self.split == 'test' and not is_training_image):
                filtered_images.append((img_id, img_name))
        
        # Get class names
        with open(os.path.join(self.dataset_directory, "classes.txt"), "r") as f:
            classes = [line.strip().split('.')[1].replace('_', '') for line in f.readlines()]

        # Generate all pairs for each class
        data = []
        for class_id in range(1, 201):
            class_images = [img for img in filtered_images if image_class_labels[img[0]] == class_id]
            for source, target in itertools.combinations(class_images, 2):
                source_image_path = os.path.join(self.images_dir, source[1])
                target_image_path = os.path.join(self.images_dir, target[1])
                
                source_points = np.array(part_locs[source[0]], dtype=float)
                target_points = np.array(part_locs[target[0]], dtype=float)

                # Filter out points that are not visible in either of the images
                visible_points = np.logical_and(source_points[:, 2], target_points[:, 2])
                source_points = torch.tensor(source_points[visible_points][:, :2], dtype=torch.float16)
                target_points = torch.tensor(target_points[visible_points][:, :2], dtype=torch.float16)

                source_bbox = torch.tensor(bounding_boxes[source[0]], dtype=torch.float16)
                target_bbox = torch.tensor(bounding_boxes[target[0]], dtype=torch.float16)
                
                # Get category from image class id
                category = classes[class_id - 1]

                data.append({
                    'source_image_path': source_image_path,
                    'target_image_path': target_image_path,
                    'source_points': source_points,
                    'target_points': target_points,
                    'source_bbox': source_bbox,
                    'target_bbox': target_bbox,
                    'source_category': category,
                    'target_category': category
                })
        
        return data


class S2K(CorrespondenceDataset):
    """
    Dataset class for the S2K dataset.
    """

    def load_data(self):
        annotation_file = os.path.join(self.dataset_directory, 's2k.json')
        with open(annotation_file, 'r') as file:
            annotations = json.load(file)

        data = []
        for a in annotations:
            source_bbox = torch.tensor(a['source_bbox'], dtype=torch.float16) # is already in (x, y, w, h) format
            target_bbox = torch.tensor(a['target_bbox'], dtype=torch.float16) # is already in (x, y, w, h) format

            data.append({
                'source_image_path': os.path.join(self.dataset_directory, "VOC2010", "JPEGImages", a['source_image_path']),
                'target_image_path': os.path.join(self.dataset_directory, "VOC2010", "JPEGImages", a['target_image_path']),
                'source_annotation_path': os.path.join(self.dataset_directory, "Annotations_Part",  a['source_annotation_path']),
                'target_annotation_path': os.path.join(self.dataset_directory, "Annotations_Part", a['target_annotation_path']),
                'source_bbox': source_bbox,
                'target_bbox': target_bbox,
                'source_parts': a['source_parts'],
                'target_parts': a['target_parts'],
                'source_category': a['source_category'],
                'target_category': a['target_category']
            })

        return data
    
    def sample_points(self, sample):
        # Load parts
        source_annotation = np.load(sample['source_annotation_path'], allow_pickle=True).item()
        target_annotation = np.load(sample['target_annotation_path'], allow_pickle=True).item()

        sample['source_annotation'] = source_annotation
        sample['target_annotation'] = target_annotation
    
        # Sample center points of parts
        source_parts = source_annotation['parts']
        target_parts = target_annotation['parts']
        if self.config.get('only_non_unique', False):
            source_parts = [part for i, part in enumerate(source_parts) if i in sample['source_parts']]
            target_parts = [part for i, part in enumerate(target_parts) if i in sample['target_parts']]

        source_points = []
        for part in source_parts:
            y, x = np.where(part["mask"])
            source_points.append([np.mean(x), np.mean(y)])
        
        target_points = []
        for part in target_parts:
            y, x = np.where(part["mask"])
            target_points.append([np.mean(x), np.mean(y)])

        sample['source_points'] = torch.tensor(source_points, dtype=torch.float16)
        sample['target_points'] = torch.tensor(target_points, dtype=torch.float16)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx]) # prevent memory leak

        # Load image
        sample['source_image'] = Image.open(sample['source_image_path'])
        sample['target_image'] = Image.open(sample['target_image_path'])

        # Save image size
        sample['source_size'] = sample['source_image'].size
        sample['target_size'] = sample['target_image'].size

        # Sample points
        self.sample_points(sample)

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample