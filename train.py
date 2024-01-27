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

def distill(teacher, student, dataloader, criterion, optimizer, num_epochs, device):
    teacher.eval()
    student.train()

    min_loss = float('inf')

    pbar = tqdm.tqdm(total=len(dataloader))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in dataloader:
            # Load images on device
            batch['image'] = batch['image'].to(device)

            # Run through model
            with torch.no_grad():
                teacher_features = teacher(batch)
            student_features = student(batch)

            # Calculate loss
            loss = criterion(teacher_features, student_features)

            # Backpropagate loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save model if loss is lowest
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(student.state_dict(), 'distilldift.pt')

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'Loss': loss.item()})

    pbar.close()

def train(model, dataloader, criterion, optimizer, num_epochs, device):
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
    num_epochs = model_config.get('num_epochs', 100)
    learning_rate = model_config.get('learning_rate', 1e-4)

    # Load model
    teacher = load_model(model_config['teacher_name'], model_config['teacher_config'], device_type)
    student = load_model(model_config['student_name'], model_config['student_config'], device_type)

    # Move model to device
    device = torch.device(device_type)
    teacher.to(device)
    student.to(device)

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)

    # Load dataset parameters
    config = dataset_config[dataset_name]
    random_sampling = config.get('random_sampling', False)

    preprocess = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)), # ImageNet mean and std
    ])
    dataset = load_dataset(dataset_name, config, preprocess)

    # Create dataloader
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=random_sampling,
                                 num_workers=num_workers)
    
    # Create criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    # Run training
    distill(teacher, student, dataloader, criterion, optimizer, num_epochs, device)

    print(f"\n{'='*30} Finished {'='*30}\n")