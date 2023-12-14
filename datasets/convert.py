# get paths from dataset_config.json
# load each dataset and turn into hdf5

import os
import h5py
import tqdm
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import read_config, load_dataset

def convert_dataset(dataset_config):
    """
    Convert a dataset and save it as an HDF5 file.
    """
    for config in dataset_config:
        # Skip datasets that do not have the 'from_hdf5' flag set to True
        if not 'from_hdf5' in config or not config['from_hdf5']:
            print(f"Dataset {config['name']} does not need converting.")
            continue
        
        # Create a new HDF5 file (stored in the same directory as the dataset, named converted.h5)
        hdf5_filename = os.path.join(config['path'], 'converted.h5')

        # check if file already exists
        if os.path.exists(hdf5_filename):
            print(f"Dataset {config['name']} already converted.")
            continue
        
        with h5py.File(hdf5_filename, 'w') as hdf5_file:
            dataset = load_dataset(config, from_hdf5=False)
            for i in tqdm.tqdm(range(len(dataset))):
                sample = dataset[i]
                image_pair = hdf5_file.create_group(str(i))
                image_pair.create_dataset('source_image', data=sample['source_image'])
                image_pair.create_dataset('target_image', data=sample['target_image'])
                image_pair.create_dataset('source_points', data=sample['source_points'])
                image_pair.create_dataset('target_points', data=sample['target_points'])
                if 'source_bbox' in sample:
                    image_pair.create_dataset('source_bbox', data=sample['source_bbox'])
                    image_pair.create_dataset('target_bbox', data=sample['target_bbox'])
        print(f"Dataset {config['name']} converted.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert datasets into HDF5.')
    parser.add_argument('--dataset_config', type=str, default='dataset_config.json', help='Path to the dataset config file.')
    args = parser.parse_args()

    # Load dataset config file
    dataset_config = read_config(args.dataset_config)

    # Convert the dataset
    convert_dataset(dataset_config)