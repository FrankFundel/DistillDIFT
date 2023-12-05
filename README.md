# DistillDIFT
Distilling the capability of large diffusion models for semantic correspondence.

# ToDo
1. Replicate Hedlin et al.: https://github.com/ubc-vision/LDM_correspondences
2. Replicate Tang et al.: https://github.com/Tsingularity/dift
3. Replicate Zhang et al.: https://github.com/Junyi42/sd-dino
4. Replicate Luo et al.: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures

# Evaluation

1. Download datasets
    - SPair-71k: `bash datasets/download_spair.sh`
    - PF-WILLOW: `bash datasets/download_pfwillow.sh`

2. (Optionally) Convert them to HDF5
    - SPair-71k: `python datasets/preprocess_spair.py --dataset_directory --hdf5_filename`
    - PF-WILLOW: `python datasets/preprocess_spair.py --dataset_directory --csv_file --hdf5_filename`

3. Run the evaluation script: `python evaluate.py [options]`

# Training

To be continued...