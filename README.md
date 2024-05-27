# DistillDIFT

Distilling the capability of large diffusion models and transformer models for semantic correspondence.
We provide a well-written and documented code base for semantic correspondence, feel free to use it!

## 🚀 Getting Started

_To be continued..._

## ⏳ ToDo
- Clean up code
- Remove private paths
- Update README.md

## 💼 Related Work

1. Hedlin et al.: https://github.com/ubc-vision/LDM_correspondences
2. Tang et al.: https://github.com/Tsingularity/dift
3. Zhang et al.: https://github.com/Junyi42/sd-dino
4. Luo et al.: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures

## 🧫 Evaluation

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

4. Run the evaluation script: `accelerate launch --multi_gpu --num_processes [n] eval.py [options]`
    - Make sure to set visible GPUs e.g. `export CUDA_VISIBLE_DEVICES=0`
    - Some models need a different diffusers version:
        - hedlin: `diffusers==0.8.0`
        - tang: `diffusers==0.15.0`
        - luo: `diffusers==0.14.0`
    - For all other models we use `diffusers==0.24.0`
    - Use `--use_cache` to speed up evaluation
    - Use `--plot` to plot the results of layerwise evaluation
    - When using cache, remember to use `--reset_cache` if you made changes on the model, else the features will stay the same

    Example: `accelerate launch --multi_gpu --num_processes 4 eval.py distilled_model --use_cache --reset_cache`

## 🔬 Training

- Supervised Training: `accelerate launch --multi_gpu --num_processes 4 train.py distilled_s --dataset_name SPair-71k`
- Weakly Supervised Distillation: `accelerate launch --multi_gpu --num_processes 4 train.py distilled_ws --dataset_name SPair-71k --use_cache`
- Unsupervised Distillation: `accelerate launch --multi_gpu --num_processes 4 train.py distilled_us --dataset_name COCO --use_cache`

For fully unsupervised distillation, retrieval-based image sampling is needed, therefore you first have to embed the dataset using the following command: `python embed.py --dataset_name COCO`

## ⚗️ Distilled models

_To be continued..._

## ⭐ Demos

- Dataset demo: `notebooks/dataset_demo.ipynb`
- Evaluation demo: `notebooks/eval_demo.ipynb`
- S2K Benchmark demo: `notebooks/s2k_demo.ipynb`
- Qualitative analysis: `notebooks/qualitative_analysis.ipynb`
- Quantitative analysis: `notebooks/quantitative_analysis.ipynb`
- Cross-Attention Maps: `notebooks/cross_attention_maps.ipynb`
