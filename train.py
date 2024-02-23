import torch
import signal
import argparse
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import torch.utils.data as data
from torch.nn.functional import interpolate
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import read_dataset_config, load_dataset, Preprocessor
from utils.model import read_model_config, load_model
from utils.correspondence import points_to_idxs, flatten_features, normalize_features, rescale_points

torch.backends.cuda.matmul.allow_tf32 = True

def distill(teacher, student, dataloader, criterion, optimizer, scheduler, num_epochs, accelerator, strategy):
    teacher.eval()
    student.train()

    epoch_loss = []
    min_loss = float('inf')

    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        pbar = tqdm(total=len(dataloader), disable=(not accelerator.is_main_process))
        for i, batch in enumerate(dataloader):
            with accelerator.accumulate(student):
                # Concatenate images
                images = torch.cat([batch['source_image'], batch['target_image']], dim=0)
                categories = batch['source_category'] + batch['target_category']

                # Run through model
                with torch.no_grad():
                    teacher_features = teacher(images, categories)
                student_features = student(images, categories)

                student_features = interpolate(student_features, teacher_features.shape[-2:], mode="bilinear") # this could eventually be replaced by a trainable upscaler
                teacher_features = interpolate(teacher_features, student_features.shape[-2:], mode="bilinear") # maybe this is better/fairer

                # Normalize features
                student_source_features = torch.nn.functional.normalize(student_features[:len(student_features) // 2], dim=-1).flatten(2)
                student_target_features = torch.nn.functional.normalize(student_features[len(student_features) // 2:], dim=-1).flatten(2)
                teacher_source_features = torch.nn.functional.normalize(teacher_features[:len(teacher_features) // 2], dim=-1).flatten(2)
                teacher_target_features = torch.nn.functional.normalize(teacher_features[len(teacher_features) // 2:], dim=-1).flatten(2)

                # Calculate similarity
                if strategy == 'full':
                    student_similarity = student_source_features.transpose(-2, -1) @ student_target_features
                    teacher_similarity = teacher_source_features.transpose(-2, -1) @ teacher_target_features
                elif strategy == 'ground_truth':
                    pass
                # bounding box, foreground, random sampling, etc.

                # Softmax
                #student_similarity = torch.nn.functional.softmax(student_similarity, dim=-1)
                #teacher_similarity = torch.nn.functional.softmax(teacher_similarity, dim=-1)

                # Calculate loss
                loss = criterion(teacher_similarity, student_similarity)

                # Backpropagate loss
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                # Logging
                accelerator.log({
                    "loss": loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=i + len(dataloader) * epoch)
                epoch_loss.append(loss.item())

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'Loss': sum(epoch_loss) / len(epoch_loss)})

                # Save model if loss is lowest
                if i > 0 and i % 100 == 0:
                    mean_loss = sum(epoch_loss) / len(epoch_loss)
                    if mean_loss < min_loss:
                        min_loss = mean_loss
                        accelerator.save_state('checkpoints/best_model', safe_serialization=False) # use model_name
        pbar.close()
    
    # Log end of training
    accelerator.end_training()

def soft_argmax(logits, beta=1.0):
    """
    Compute the soft-argmax of a tensor.
    
    Parameters:
    - logits: A tensor of logits.
    - beta: A scalar controlling the sharpness of the output distribution. Higher values make the distribution closer to argmax.
    
    Returns:
    - A tensor representing the soft-argmax of the input.
    """
    weights = torch.softmax(beta * logits, dim=-1)
    indices = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
    soft_argmax_val = torch.sum(weights * indices, dim=-1)
    return soft_argmax_val

class CLIPContrastiveLoss(nn.Module):
    def __init__(self, tau=0.07):
        super(CLIPContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, similarity_matrix):
        similarity_matrix = similarity_matrix / self.tau

        # Generate labels for matching pairs
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Compute loss for image-to-text and text-to-image
        loss_i2t = F.cross_entropy(similarity_matrix.transpose(0, 1), labels, reduction="mean")
        loss_t2i = F.cross_entropy(similarity_matrix, labels, reduction="mean")

        # Return the average of both losses
        return (loss_i2t + loss_t2i) / 2

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets.argmax(dim=1))

        # Reverse cross-entropy loss
        pred_prob = F.softmax(predictions, dim=1)
        rce_loss = -torch.mean(torch.sum(targets * torch.log(pred_prob + 1e-7), dim=1))

        # Symmetric loss
        return self.alpha * ce_loss + self.beta * rce_loss
    
def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, accelerator, similarity_method):
    model.train()

    epoch_loss = []
    min_loss = float('inf')

    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        pbar = tqdm(total=len(dataloader), disable=(not accelerator.is_main_process))
        for i, batch in enumerate(dataloader):
            with accelerator.accumulate(student):
                # Concatenate images
                images = torch.cat([batch['source_image'], batch['target_image']], dim=0)
                categories = batch['source_category'] + batch['target_category']
                source_points = batch['source_points']
                target_points = batch['target_points']

                # Run through model
                features = model(images, categories)

                features = interpolate(features, features.shape[-2:], mode="bilinear") # this could eventually be replaced by a trainable upscaler
                
                # Normalize features
                source_features = features[:len(features) // 2]
                target_features = features[len(features) // 2:]

                # Prepare points
                h, w = source_features.shape[-2:]
                source_points = rescale_points(source_points, images.shape[-2:], (h, w)) # [B, N, 2]
                source_idxs = points_to_idxs(source_points, (h, w)) # [B, N]
                target_points = rescale_points(target_points, images.shape[-2:], (h, w)) # [B, N, 2]
                target_idxs = points_to_idxs(target_points, (h, w)) # [B, N]

                # Use source points to get features
                source_features = flatten_features(source_features) # [B, HxW, C]
                source_features = normalize_features(source_features) # [B, HxW, C]
                source_features = source_features[torch.arange(source_features.shape[0])[:, None], source_idxs] # [B, N, C]
                
                # Calculate similarity map
                target_features = flatten_features(target_features) # [B, HxW, C]
                target_features = normalize_features(target_features) # [B, HxW, C]
                similarity_map = source_features @ target_features.transpose(1, 2) # [B, N, HxW]

                if similarity_method == 'softmax': # cross-entropy
                    prediction = torch.nn.functional.softmax(similarity_map, dim=-1) # [B, N, HxW]
                    target = torch.nn.functional.one_hot(target_idxs, num_classes=similarity_map.shape[-1]) # [B, N, HxW]
                    # TODO: perturb target
                elif similarity_method == 'soft_argmax': # MSE
                    prediction = soft_argmax(similarity_map, beta=1.0) # [B, N]
                    target = target_idxs # [B, N]
                    # TODO: perturb target
                elif similarity_method == 'contrastive': # CLIP symmetric contrastive loss
                    prediction = source_features # [B, N, C]
                    target = target_features[torch.arange(target_features.shape[0])[:, None], target_idxs] # [B, N, C]

                # Calculate loss
                loss = criterion(prediction, target)

                # Backpropagate loss
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                # Logging
                accelerator.log({
                    "loss": loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=i + len(dataloader) * epoch)
                epoch_loss.append(loss.item())

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'Loss': sum(epoch_loss) / len(epoch_loss)})

                # Save model if loss is lowest
                if i > 0 and i % 100 == 0:
                    mean_loss = sum(epoch_loss) / len(epoch_loss)
                    if mean_loss < min_loss:
                        min_loss = mean_loss
                        accelerator.save_state('checkpoints/best_model', safe_serialization=False) # use model_name
        pbar.close()
    
    # Log end of training
    accelerator.end_training()

def signal_handler(accelerator):
    def fn(signal, frame):
        print('Signal was sent, stopping training...')
        accelerator.save_state()
        accelerator.end_training()
        exit(0)
    return fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of model to train')
    parser.add_argument('--dataset_name', type=str, default='ImageNet', help='Name of dataset to train on')
    parser.add_argument('--model_config', type=str, default='train_config.yaml', help='Path to model config file')
    parser.add_argument('--dataset_config', type=str, default='dataset_config.yaml', help='Path to dataset config file')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    model_config = args.model_config
    dataset_config = args.dataset_config
    num_workers = args.num_workers
    checkpoint = args.checkpoint

    # Load model config
    model_config = read_model_config(model_config)[model_name]

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)
    num_epochs = model_config.get('num_epochs', 100)
    learning_rate = model_config.get('learning_rate', 1e-4)
    gradient_accumulation_steps = model_config.get('gradient_accumulation_steps', 1)
    mode = model_config.get('mode', 'train')
    strategy = model_config.get('strategy', 'full')
    similarity_method = model_config.get('similarity_method', 'softmax')
    loss_function = model_config.get('loss_function', 'cross_entropy')

    # Load model(s)
    if mode == 'distill':
        teacher = load_model(model_config['teacher_name'], model_config['teacher_config'])
        student = load_model(model_config['student_name'], model_config['student_config'])
    elif mode == 'train':
        student = load_model(model_name, model_config)

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)

    # Load dataset parameters
    config = dataset_config[dataset_name]
    random_sampling = config.get('random_sampling', False)

    preprocess = Preprocessor(image_size, rescale_data=True, normalize_image=(mode == 'distill'))
    dataset = load_dataset(dataset_name, config, preprocess)
    dataset.image_pair = True

    # Create dataloader
    def collate_fn(batch):
        output = {}
        for key in batch[0].keys():
            output[key] = [sample[key] for sample in batch]
            if key in ['source_image', 'target_image']:
                output[key] = torch.stack(output[key])
        return output

    # Create dataloader
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=random_sampling,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    
    # Create criterion and optimizer
    learning_rate = gradient_accumulation_steps * learning_rate
    params = student.params_to_optimize if hasattr(student, "params_to_optimize") else student.parameters()
    if loss_function == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_function == 'contrastive':
        criterion = CLIPContrastiveLoss()
    elif loss_function == 'symmetric':
        criterion = SymmetricCrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    accelerator = Accelerator(log_with="tensorboard", project_config=ProjectConfiguration(
        project_dir=".",
        logging_dir="logs"
    ),
    mixed_precision="bf16",
    gradient_accumulation_steps=gradient_accumulation_steps)
    tracker_config = {
        "dataset_name": dataset_name,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "image_size": str(image_size),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mode": mode,
        "strategy": strategy,
        "similarity_method": similarity_method,
        "loss_function": loss_function,
        # scheduler etc.
    }
    if mode == 'distill':
        tracker_config["teacher_name"] = model_config['teacher_name']
        tracker_config["student_name"] = model_config['student_name']
        teacher, student, optimizer, scheduler, dataloader = accelerator.prepare(teacher, student, optimizer, scheduler, dataloader)
        accelerator.register_for_checkpointing(student) #, optimizer, scheduler)
    elif mode == 'train':
        tracker_config["model_name"] = model_name
        student, optimizer, scheduler, dataloader = accelerator.prepare(student, optimizer, scheduler, dataloader)
        accelerator.register_for_checkpointing(student) #, optimizer, scheduler)

    accelerator.init_trackers(model_name, config=tracker_config)
    if checkpoint is not None:
        accelerator.load_state(checkpoint)

    # Print seperator
    if accelerator.is_main_process:
        print(f"\n{'='*30} Training {model_name} {'='*30}\n")
    
    signal.signal(signal.SIGTERM, signal_handler)

    # Run training
    if mode == 'distill':
        distill(teacher, student, dataloader, criterion, optimizer, scheduler, num_epochs, accelerator, strategy)
    elif mode == 'train':
        train(student, dataloader, criterion, optimizer, scheduler, num_epochs, accelerator, similarity_method)

    if accelerator.is_main_process:
        print(f"\n{'='*30} Finished {'='*30}\n")