# DistillDIFT
Distilling the capability of large diffusion models for semantic correspondence.

# ToDo
- DistributedDataParallel
- create evaluation notebook for single samples and visualization

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

3. (Optionally) Convert datasets to HDF5: `python datasets/convert.py --dataset_config`

4. Run the evaluation script: `python evaluate.py [options]`
    - Make sure to do set visible GPUs e.g. `export CUDA_VISIBLE_DEVICES=0,1`

# Training

_To be continued..._

# Demos

- Dataset exploration: `datasets/explore.ipynb`