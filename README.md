# DistillDIFT
Distilling the capability of large diffusion models for semantic correspondence.
We provide a well-written and documented code base for semantic correspondence, feel free to use it!

## ToDo
- Implement accelerate for multiple GPUs and nodes: https://www.youtube.com/watch?v=s7dy8QRgjJ0
- Implement sharded dataset
- Implement different training strategies
- Let use_cache be defined by model class

## Related Work
1. Hedlin et al.: https://github.com/ubc-vision/LDM_correspondences
2. Tang et al.: https://github.com/Tsingularity/dift
3. Zhang et al.: https://github.com/Junyi42/sd-dino
4. Luo et al.: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures

## Evaluation

### Tutorial

1. Download datasets
    - SPair-71k: `bash datasets/download_spair.sh`
    - PF-WILLOW: `bash datasets/download_pfwillow.sh`
    - CUB-200-2011: `bash datasets/download_cub.sh`

2. Setup your `dataset_config.yaml`
    - `path`: absolute path to the dataset
    - `num_samples`: limit the number of samples to use
    - `random_sampling`: whether to shuffle the dataset before sampling

3. Setup your `eval_config.yaml`
    - `image_size`: size of the input images after resizing
    - `batch_size`: overwrite the batch size for evaluation
    - `grad_enabled`: whether to enable gradient calculation
    - `drop_last_batch`: whether to drop the last batch if it is smaller than the batch size
    - `layers`: list of layers to evaluate, only possible together with `--use_cache`
    - Additional options are passed to the model

4. Run the evaluation script: `python eval.py [options]`
    - Make sure to do set visible GPUs e.g. `export CUDA_VISIBLE_DEVICES=0`
    - Some models need a different diffusers version:
        - hedlin: `diffusers==0.8.0`
        - tang: `diffusers==0.15.0`
        - luo: `diffusers==0.14.0`
    - For all other models we use `diffusers==0.24.0`
    - Use `--use_cache` to speed up evaluation
    - Use `--plot` to plot the results of layerwise evaluation
    - When using cache, remember to use `--reset_cache` the cache file if you made changes on the model, else the features will stay the same

## Training

_To be continued..._

## Distilled models

_To be continued..._

## Demos

- Dataset demo: `notebooks/dataset_demo.ipynb`
- Evaluation demo: `notebooks/eval_demo.ipynb`
