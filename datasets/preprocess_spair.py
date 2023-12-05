import os
import json
import h5py
import tqdm
import numpy as np
from PIL import Image
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Preprocess SPair-71k dataset.')
parser.add_argument('--dataset_directory', type=str, default='/export/group/datasets/SPair-71k', help='Path to the directory containing the SPair-71k dataset.')
parser.add_argument('--hdf5_filename', type=str, default='/export/group/datasets/SPair-71k/spair71k.h5', help='Name of the HDF5 file to be created.')
args = parser.parse_args()

def get_filenames(directory, extension=''):
    return [f for f in os.listdir(directory) if f.endswith(extension) and not f.startswith('.')]

def preprocess(dataset_directory, hdf5_filename):
    """
    Create an HDF5 file from the SPair-71k dataset.

    :param dataset_directory: Path to the directory containing the SPair-71k dataset.
    :param hdf5_filename: Name of the HDF5 file to be created.
    """

    images_directory = os.path.join(dataset_directory, 'JPEGImages')
    annotations_directory = os.path.join(dataset_directory, 'PairAnnotation/test')

    # Create a new HDF5 file
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        # Read annotations
        annotations_filenames = get_filenames(annotations_directory, '.json')
        for annotations_filename in tqdm.tqdm(annotations_filenames):
            # read annotation
            with open(os.path.join(annotations_directory, annotations_filename), 'r') as f:
                annotation = json.load(f)
            
            category = annotation['category']
            source_image = annotation['src_imname']
            target_image = annotation['trg_imname']
            source_points = annotation['src_kps']
            target_points = annotation['trg_kps']
            source_bbox = annotation['src_bndbox']
            target_bbox = annotation['trg_bndbox']

            # Create a group for the image pair
            group = hdf5_file.create_group(source_image + '_' + target_image)

            # Load images
            source_image = Image.open(os.path.join(images_directory, category, source_image))
            target_image = Image.open(os.path.join(images_directory, category, target_image))

            # Create datasets for images
            group.create_dataset('source_image', data=np.array(source_image))
            group.create_dataset('target_image', data=np.array(target_image))

            # Create dataset for correspondence points
            group.create_dataset('source_points', data=np.array(source_points))
            group.create_dataset('target_points', data=np.array(target_points))

            # Create dataset for bounding boxes
            group.create_dataset('source_bbox', data=np.array(source_bbox))
            group.create_dataset('target_bbox', data=np.array(target_bbox))


# Preprocess SPair-71k dataset
if __name__ == '__main__':
    preprocess(args.dataset_directory, args.hdf5_filename)

