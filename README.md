# DistillDIFT
Distilling the capability of large diffusion models for semantic correspondence.
We provide a well-written and documented code base for semantic correspondence, feel free to use it!

# ToDo
- Implement layer- and timestep- wise PCK
    - DiffusionModel returns nested list of timesteps, layers and predicted points
- DistributedDataParallel

- Maybe remove converter and ConvertedDataset if no performance gain
- Make code simpler and more beautiful
- Try not loading onto GPU when cached
- Try not resizing to image_size but to source_size and target_size when calculating correspondence

- Make wrapper for Dataset instead
- Zhang with batch mode
- model_config.json with load_model() function
- discription for argparse

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
    - If you want it to be converted and loaded as HDF5, set the flag `from_hdf5` to true
    - If you want to limit the number of samples for a specific dataset, set `num_samples` to an arbitrary integer

3. (Optionally) Convert datasets to HDF5: `python datasets/convert.py --dataset_config`

4. Run the evaluation script: `python evaluate.py [options]`
    - Make sure to do set visible GPUs e.g. `export CUDA_VISIBLE_DEVICES=0`
    - Some models need a different diffusers version:
        - hedlin: `diffusers==0.8.0`
        - tang: `diffusers==0.15.0`
        - luo: `diffusers==0.14.0`
    - For all other models we use `diffusers==0.24.0`
    - When using cache, remember to delete the cache file if you made changes on the model, else the features will stay the same

# Training

_To be continued..._

# Feature Extractors

_To be continued..._

# Demos

- Dataset exploration: `datasets/explore.ipynb`
- Evaluation demo: `eval_demo.ipynb`
