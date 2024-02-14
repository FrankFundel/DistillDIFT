# DistillDIFT

Distilling the capability of large diffusion models and transformer models for semantic correspondence.
We provide a well-written and documented code base for semantic correspondence, feel free to use it!

## 🚀 Getting Started

_To be continued..._

## ⏳ ToDo

- Make branch ready for merging

### Analysis
1. Measure accuracy of SD and DINO using boxplot on distances
2. Create disentangled dataset using PASCAL-Part
3. Measure spatial and semantic knowledge of SD and DINO
4. Recreate qualitative analysis of layers/timesteps with segmented images and concatenated features

SD strengths:
- Spatial knowledge and semantic knowledge: Performance on disentangled dataset, semantic flow and 3D data

DINO strengths:
- Higher acc because of resolution: Performance drops on lower resolution

### Training
1. Implement no-teacher train loop
2. Train supervised on SPair-71k (no-teacher):
    1. SD, fine-tune, cross-entropy loss on softmax
    2. SD, fine-tune, MSE
    3. DINO, fine-tune, cross-entropy loss on softmax
    4. DINO, fine-tune, MSE
    5. SD, LoRA, cross-entropy loss on softmax
    6. SD, LoRA, MSE
    7. DINO, LoRA, cross-entropy loss on softmax
    8. DINO, LoRA, MSE
    9. SD, no LoRA, no fine-tune, only Upscaler, cross-entropy loss on softmax
    10. SD, no LoRA, no fine-tune, only Upscaler, MSE
    11. (DINO, no LoRA, no fine-tune, only Upscaler, cross-entropy loss on softmax)
    12. (DINO, no LoRA, no fine-tune, only Upscaler, MSE)
3. Implement strategies: Foreground and Random Point Sampling
3. Train unsupervised on ImageNet (with teacher):
    1. Best performing model from 2. on foreground with random point sampling, cross-entropy loss on softmax
    2. Best performing model from 2. on foreground with random point sampling, MSE
    3. Best performing model from 2. on foreground, MSE
    4. Best performing model from 2. on full, MSE

- Make sure there are no unnecessary gradients and everything is in eval mode and fp16
- Use kernel softmax for cross-entropy loss
- Apply LoRA to different layers
- Cache features from teacher
- Implement sharded dataset

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

4. Run the evaluation script: `python eval.py [options]`
    - Make sure to set visible GPUs e.g. `export CUDA_VISIBLE_DEVICES=0`
    - Some models need a different diffusers version:
        - hedlin: `diffusers==0.8.0`
        - tang: `diffusers==0.15.0`
        - luo: `diffusers==0.14.0`
    - For all other models we use `diffusers==0.24.0`
    - Use `--use_cache` to speed up evaluation
    - Use `--plot` to plot the results of layerwise evaluation
    - When using cache, remember to use `--reset_cache` if you made changes on the model, else the features will stay the same

## 🔬 Training

- Using accelerate: `accelerate launch --multi_gpu --num_processes 2 train.py distilldift_1`

## ⚗️ Distilled models

_To be continued..._

## ⭐ Demos

- Dataset demo: `notebooks/dataset_demo.ipynb`
- Evaluation demo: `notebooks/eval_demo.ipynb`
- Qualitative analysis: `notebooks/qualitative_analysis.ipynb`
