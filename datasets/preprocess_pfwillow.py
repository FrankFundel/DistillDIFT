import os
import h5py
import tqdm
import numpy as np
from PIL import Image
import argparse

# Parse command row arguments
parser = argparse.ArgumentParser(description='Preprocess PF-Willow dataset.')
parser.add_argument('--dataset_directory', type=str, default='../data/PF-WILLOW', help='Path to the directory containing the PF-Willow dataset.')
parser.add_argument('--csv_file', type=str, default='../data/PF-WILLOW/test_pairs_pf.csv', help='Path to the .csv file containing image pairs and correspondence points.')
parser.add_argument('--hdf5_filename', type=str, default='../data/PF-WILLOW/pfwillow.h5', help='Name of the HDF5 file to be created.')
args = parser.parse_args()

def preprocess(dataset_directory, csv_file, hdf5_filename):
    """
    Create an HDF5 file from the PF-Willow dataset.

    :param dataset_directory: Path to the directory containing the PF-Willow dataset.
    :param csv_file: Path to the .csv file containing image pairs and correspondence points.
    :param hdf5_filename: Name of the HDF5 file to be created.
    """

    # Create a new HDF5 file
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
       # Read csv file
        with open(csv_file, 'r') as f:
            rows = f.readlines()
        # skip header
        rows = rows[1:]

        # Create a group for each image pair
        for row in tqdm.tqdm(rows):
            # Get image pair and correspondence points
            row = row.split(',')
            source_image = row[0].replace('PF-dataset/', '')
            target_image = row[1].replace('PF-dataset/', '')
            source_points = list(zip(row[2:12], row[12:22])) # X, Y
            target_points = list(zip(row[22:32], row[32:])) # X, Y

            # Create a group for the image pair
            group = hdf5_file.create_group(source_image.replace('/', '_') + '_' + target_image.replace('/', '_'))

            # Load images
            source_image = Image.open(os.path.join(dataset_directory, source_image))
            target_image = Image.open(os.path.join(dataset_directory, target_image))

            # Create datasets for images
            group.create_dataset('source_image', data=np.array(source_image))
            group.create_dataset('target_image', data=np.array(target_image))

            # Create dataset for correspondence points
            group.create_dataset('source_points', data=np.array(source_points).astype(np.float32))
            group.create_dataset('target_points', data=np.array(target_points).astype(np.float32))


# Preprocess PF-Willow dataset
if __name__ == '__main__':
    preprocess(args.dataset_directory, args.csv_file, args.hdf5_filename)
