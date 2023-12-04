import os
import h5py
from PIL import Image
from scipy.io import loadmat

def create_hdf5_from_pf_willow(dataset_directory, hdf5_filename):
    """
    Create an HDF5 file from the PF-Willow dataset containing pairs of images (image_a and image_b)
    and their corresponding annotation points.

    :param dataset_directory: Path to the directory containing the PF-Willow dataset.
    :param hdf5_filename: Name of the HDF5 file to be created.
    """

    # Create a new HDF5 file
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        # Iterate over categories
        for category in [f for f in os.listdir(dataset_directory) if not f.startswith('.')]:
            category_path = os.path.join(dataset_directory, category)

            # for each file (sorted)
            # get filepaths for a and b, png and mat
            # read image with PIL
            # read mat with scipy.io.loadmat
            # save image pair and points in hdf5 file


# Example usage
create_hdf5_from_pf_willow('../../data/PF-WILLOW', 'pfwillow.hdf5')
