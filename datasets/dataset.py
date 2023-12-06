import torch
import torch.utils.data as data
import h5py
import json
import os
import csv
import numpy as np
from PIL import Image

class PreprocessedDataset(data.Dataset):
    """
    Dataset class for preprocessed datasets.
    """

    def __init__(self, hdf5_filename, transform=None):
        super(PreprocessedDataset, self).__init__()

        # Open the HDF5 file
        self.hdf5_file = h5py.File(hdf5_filename, 'r')

        # Get the image pairs
        self.image_pairs = list(self.hdf5_file.keys())

        # Store the transform
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        # Get image pair
        pair_key = self.image_pairs[index]
        image_pair = self.hdf5_file[pair_key]

        # Get the source image, target image, correspondence points and bounding boxes (if they exist)
        source_image = Image.fromarray(image_pair['source_image'][()])
        target_image = Image.fromarray(image_pair['target_image'][()])
        source_points = image_pair['source_points'][()]
        target_points = image_pair['target_points'][()]
        if 'source_bbox' in image_pair:
            source_bbox = image_pair['source_bbox'][()]
            target_bbox = image_pair['target_bbox'][()]

        # Apply the transform to the images if requested
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        # Return the image pair, correspondence points and bounding boxes (if they exist)
        sample = {
            'source_image': source_image,
            'target_image': target_image,
            'source_points': source_points,
            'target_points': target_points
        }
        if 'source_bbox' in image_pair:
            sample['source_bbox'] = source_bbox
            sample['target_bbox'] = target_bbox
        return sample


class PFWillowDataset(data.Dataset):
    """
    Dataset class for the PF-Willow dataset.
    """

    def __init__(self, dataset_directory, csv_file, transform=None):
        self.dataset_directory = dataset_directory
        self.csv_file = csv_file
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                source_image_path = os.path.join(self.dataset_directory, row[0].replace('PF-dataset/', ''))
                target_image_path = os.path.join(self.dataset_directory, row[1].replace('PF-dataset/', ''))

                source_points = np.array(list(zip(row[2:12], row[12:22]))).astype(np.float16) # X, Y
                target_points = np.array(list(zip(row[22:32], row[32:]))).astype(np.float16) # X, Y

                data.append((source_image_path, target_image_path, source_points, target_points))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_image_path, target_image_path, source_points, target_points = self.data[idx]

        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return {
            'source_image': source_image,
            'target_image': target_image,
            'source_points': source_points,
            'target_points': target_points
        }


class SPairDataset(data.Dataset):
    """
    Dataset class for the SPair-71k dataset.
    """

    def __init__(self, dataset_directory, transform=None):
        self.images_dir = os.path.join(dataset_directory, 'JPEGImages')
        self.annotations_dir = os.path.join(dataset_directory, 'PairAnnotation/test')
        self.transform = transform

        # Load annotation filenames
        self.annotations_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.json')]
        self.data = self.load_data()

    def __len__(self):
        return len(self.annotations_files)

    def load_data(self):
        data = []
        for annotation_file in self.annotations_files:
            with open(os.path.join(self.annotations_dir, annotation_file), 'r') as file:
                annotation = json.load(file)

            category = annotation['category']
            source_image_path = os.path.join(self.images_dir, category, annotation['src_imname'])
            target_image_path = os.path.join(self.images_dir, category, annotation['trg_imname'])
            source_points = np.array(annotation['src_kps']).astype(np.float16)
            target_points = np.array(annotation['trg_kps']).astype(np.float16)
            source_bbox = np.array(annotation['src_bndbox']).astype(np.float16)
            target_bbox = np.array(annotation['trg_bndbox']).astype(np.float16)

            data.append((source_image_path, target_image_path, source_points, target_points, source_bbox, target_bbox))
        return data

    def __getitem__(self, idx):
        source_image_path, target_image_path, source_points, target_points, source_bbox, target_bbox = self.data[idx]

        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return {
            'source_image': source_image,
            'target_image': target_image,
            'source_points': source_points,
            'target_points': target_points,
            'source_bbox': source_bbox,
            'target_bbox': target_bbox
        }
