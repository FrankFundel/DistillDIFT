import torch
from accelerate.utils import tqdm # import tqdm
import argparse

from accelerate import Accelerator

import torch.utils.data as data
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.nn.functional import interpolate

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

def distill(teacher, student, dataloader, criterion, optimizer, num_epochs, device, accelerator):
    teacher.eval()
    student.train()

    epoch_loss = []
    min_loss = float('inf')

    pbar = tqdm(total=len(dataloader))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in enumerate(dataloader):
            # Load images on device
            batch['source_image'] = batch['source_image'].to(device)
            batch['target_image'] = batch['target_image'].to(device)
            
            # Concatenate images
            images = torch.cat([batch['source_image'], batch['target_image']], dim=0)
            categories = batch['category'] * 2

            # Run through model
            with torch.no_grad():
                teacher_features = teacher(images, categories)
            student_features = student(images, categories)

            student_features = interpolate(student_features, teacher_features.shape[-2:]) # this could eventuall be replaced by a trainable upscaler

            # Normalize and calcuate dot product between source and target features
            student_source_features = torch.nn.functional.normalize(student_features[:len(student_features) // 2], dim=-1).flatten(2)
            student_target_features = torch.nn.functional.normalize(student_features[len(student_features) // 2:], dim=-1).flatten(2)
            teacher_source_features = torch.nn.functional.normalize(teacher_features[:len(teacher_features) // 2], dim=-1).flatten(2)
            teacher_target_features = torch.nn.functional.normalize(teacher_features[len(teacher_features) // 2:], dim=-1).flatten(2)

            student_similarity = student_source_features.transpose(-2, -1) @ student_target_features
            teacher_similarity = teacher_source_features.transpose(-2, -1) @ teacher_target_features

            # Calculate loss
            loss = criterion(teacher_similarity, student_similarity)
            epoch_loss.append(loss.item())

            # Backpropagate loss
            optimizer.zero_grad()
            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'Loss': sum(epoch_loss) / len(epoch_loss)})

            # Save model if loss is lowest
            if i % 50 == 0:
                mean_loss = sum(epoch_loss) / len(epoch_loss)
                if mean_loss < min_loss:
                    min_loss = mean_loss
                    torch.save(student.state_dict(), 'distilldift.pt')

    pbar.close()

def train(model, dataloader, criterion, optimizer, num_epochs, device):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of model to train')
    parser.add_argument('--dataset_name', type=str, default='ImageNet', help='Name of dataset to train on')
    parser.add_argument('--model_config', type=str, default='train_config.yaml', help='Path to model config file')
    parser.add_argument('--dataset_config', type=str, default='dataset_config.yaml', help='Path to dataset config file')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    model_config = args.model_config
    dataset_config = args.dataset_config
    num_workers = args.num_workers

    # Load model config
    model_config = read_model_config(model_config)[model_name]

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)
    num_epochs = model_config.get('num_epochs', 100)
    learning_rate = model_config.get('learning_rate', 1e-4)

    # Load model
    accelerator = Accelerator()
    device = accelerator.device

    teacher = load_model(model_config['teacher_name'], model_config['teacher_config'])
    student = load_model(model_config['student_name'], model_config['student_config'])

    # Move model to device
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
    dataset.image_pair = True

    # Create dataloader
    def collate_fn(batch):
        return {
            'source_image': torch.stack([sample['source_image'] for sample in batch]),
            'target_image': torch.stack([sample['target_image'] for sample in batch]),
            'category': [sample['category'] for sample in batch],
            'source_size': [sample['source_size'] for sample in batch],
            'target_size': [sample['target_size'] for sample in batch],
        }

    # Create dataloader
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=random_sampling,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    
    # Create criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(student.params_to_optimize, lr=learning_rate)
    #optimizer = torch.optim.Adam(student.extractor.pipe.unet.parameters(), lr=learning_rate)

    teacher, student, optimizer, dataloader = accelerator.prepare(teacher, student, optimizer, dataloader)

    # Run training
    distill(teacher, student, dataloader, criterion, optimizer, num_epochs, device, accelerator)

    print(f"\n{'='*30} Finished {'='*30}\n")