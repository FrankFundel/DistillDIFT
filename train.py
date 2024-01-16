import os
import torch
import tqdm
import argparse

import torch.utils.data as data
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

from utils.dataset import read_dataset_config, load_dataset
from utils.model import read_model_config, load_model

# This python file is used to distill various models into a single smaller model.
# It should be able to:
# 1. Load a teacher model
# 2. Load a student model
# 3. Load a dataset (e.g. ImageNet)
# 4. Perform distillation:
#       1. Run each image pair through teacher and student
#       2. Compute loss between features and correspondences (e.g. MSE)
#       3. Backpropagate loss

# TODO: Add support for multiple layers
# TODO: Add support for multiple timesteps
# TODO: Add support for multiple teachers
# TODO: Add support for fine-tuning

def distill(teacher, student, dataloader, criterion, optimizer, device):
    pass

def train(model, dataloader, criterion, optimizer, device):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of model to train')
    parser.add_argument('--dataset_name', type=str, default='ImageNet', help='Name of dataset to train on')
    parser.add_argument('--model_config', type=str, default='train_config.yaml', help='Path to model config file')
    parser.add_argument('--dataset_config', type=str, default='dataset_config.yaml', help='Path to dataset config file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    model_config = args.model_config
    dataset_config = args.dataset_config
    device_type = args.device
    num_workers = args.num_workers

    # Load model config
    model_config = read_model_config(model_config)[model_name]

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)

    # Load model
    model = load_model(model_name, model_config, device_type)

    # Move model to device
    device = torch.device(device_type)
    model.to(device)

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)

    # Load dataset parameters
    config = dataset_config[dataset_name]
    random_sampling = config.get('random_sampling', False)

    preprocess = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = load_dataset(dataset_name, config, preprocess)

    # Create dataloader
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=random_sampling,
                                 num_workers=num_workers)
    
    # Create criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run training
    distill(model, dataloader, criterion, optimizer, device)

    print(f"\n{'='*30} Finished {'='*30}\n")