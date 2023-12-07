# DistillDIFT
Distilling the capability of large diffusion models for semantic correspondence.

# ToDo
- use float16 and convert('RGB) in convert and dataset class
- make model loading function
- manage load_size and rescale points
- adaptable batch_size for luo

# Related Work
1. Hedlin et al.: https://github.com/ubc-vision/LDM_correspondences
2. Tang et al.: https://github.com/Tsingularity/dift
3. Zhang et al.: https://github.com/Junyi42/sd-dino
4. Luo et al.: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures

# Evaluation

1. Download datasets
    - SPair-71k: `bash datasets/download_spair.sh`
    - PF-WILLOW: `bash datasets/download_pfwillow.sh`
    - CUB-200-2011: `bash datasets/download_cub.sh`

2. Setup your `dataset_config.json`
    - Use absolute path to folder
    - If you want it to be converted and loaded as HDF5 set the flag `from_hdf5` to true

2. (Optionally) Convert them to HDF5: `python datasets/convert.py --dataset_config`

3. Run the evaluation script: `python evaluate.py [options]`

# Training

_To be continued..._

# Demos

- Dataset exploration: `datasets/explore.ipynb`