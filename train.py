import os
import torch
import signal
import argparse
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import torch.utils.data as data
import torch.nn.functional as F
from torch.nn.functional import interpolate, one_hot
import torchvision

from utils.dataset import read_dataset_config, load_dataset, cache_dataset, Preprocessor
from utils.model import read_model_config, load_model
from utils.correspondence import points_to_idxs, idxs_to_points, flatten_features, normalize_features, rescale_points, compute_pck_img, compute_pck_bbox
from utils.distillation import softmax_with_temperature, softargmax2d, sample_points, separate_foreground_copca

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import DistributedDataParallelKwargs

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Source Image -> Target Image
        ce = self.cross_entropy(pred, target) # dim 0

        # Target Image -> Source Image
        rce = (-1*torch.sum(pred * torch.log(target), dim=1)) # dim 1

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
def distill_epoch(teacher, student, dataloader, criterion, optimizer, scheduler, epoch, accelerator, use_cache, sampling_method, softmax_temperature=0.01, softargmax_beta=1000.0, model_name='best_model'):
    teacher.eval()
    student.train()

    epoch_loss = []
    pbar = tqdm(total=len(dataloader), disable=(not accelerator.is_main_process))
    for i, batch in enumerate(dataloader):
        # Concatenate images
        images = torch.cat([batch['source_image'], batch['target_image']], dim=0)
        categories = batch['source_category'] + batch['target_category']

        if use_cache:
            # Load features from cache
            teacher_features = torch.cat([batch['source_features'], batch['target_features']])
        else:
            # Run through model
            with torch.no_grad():
                teacher_features = teacher(images, categories)
        student_features = student(images, categories)

        # Interpolate teacher features for higher point density
        if similarity_method == 'soft_argmax':
            teacher_features = interpolate(teacher_features, images.shape[-2:], mode="bilinear")
        else:
            teacher_features = interpolate(teacher_features, student_features.shape[-2:], mode="bilinear")
        
        # Prepare features
        _, SC, SH, SW = student_features.shape
        _, TC, TH, TW = teacher_features.shape
        B = student_features.shape[0] // 2
        student_source_features = normalize_features(flatten_features(student_features[:B])) # [B, HxW, C]
        student_target_features = normalize_features(flatten_features(student_features[B:])) # [B, HxW, C]
        teacher_source_features = normalize_features(flatten_features(teacher_features[:B])) # [B, HxW, C]
        teacher_target_features = normalize_features(flatten_features(teacher_features[B:])) # [B, HxW, C]

        # Sample points
        if sampling_method == 'full': # Full dot product
            pass # N = HxW
        elif sampling_method == 'foreground_stopgrad': # Full dot product, but only backprop through the foreground
            source_masks, target_masks = separate_foreground_copca(student_source_features, student_target_features)
            idxs = source_masks.long().nonzero(as_tuple=False)[:, 1].unsqueeze(0)
            student_source_features[torch.arange(B)[:, None], idxs].detach()
            idxs = target_masks.long().nonzero(as_tuple=False)[:, 1].unsqueeze(0)
            student_target_features[torch.arange(B)[:, None], idxs].detach()
        else:
            idxs, points = sample_points(student_source_features, (B, SC, SH, SW), sampling_method,
                                            batch["source_points"], images.shape[-2:]) # [B, N], [B, N, 2]
            student_source_features = student_source_features[torch.arange(B)[:, None], idxs]
            points = rescale_points(points, (SH, SW), (TH, TW)) # [B, N, 2]
            idxs = points_to_idxs(points, (TH, TW)) # [B, N]
            teacher_source_features = teacher_source_features[torch.arange(B)[:, None], idxs]
            
        # Calculate similarity
        student_similarity = student_source_features @ student_target_features.transpose(1, 2) # [B, N, HxW]
        teacher_similarity = teacher_source_features @ teacher_target_features.transpose(1, 2) # [B, N, HxW]
        
        # Calculate prediction and target
        if similarity_method == 'softmax': # should be combined with cross-entropy
            prediction = softmax_with_temperature(student_similarity, softmax_temperature) # [B, N, HxW]
            prediction = prediction.reshape(*prediction.shape[:2], SH, SW) # [B, N, H, W]
            target = softmax_with_temperature(teacher_similarity, softmax_temperature) # [B, N, HxW]
            target = target.reshape(*target.shape[:2], TH, TW) # [B, N, H, W]
        elif similarity_method == 'soft_argmax': # should be combined with MSE
            epsilon = 1.0 # shift target points [epsilon, -epsilon] pixels
            student_similarity = student_similarity.reshape(*student_similarity.shape[:2], SH, SW) # [B, N, H, W]
            prediction = softargmax2d(student_similarity, softargmax_beta) # [B, N, 2]
            target = torch.argmax(teacher_similarity, dim=-1).float() # [B, N]
            target = idxs_to_points(target, (TH, TW)) # [B, N, 2]
            target = rescale_points(target, (TH, TW), (SH, SW))
            target += torch.randn_like(target) * epsilon
        elif similarity_method == 'raw':
            prediction = student_similarity.reshape(*student_similarity.shape[:2], SH, SW)
            target = teacher_similarity.reshape(*teacher_similarity.shape[:2], TH, TW)

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
        if i > 0 and i % int(len(dataloader) * 0.1) == 0:
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            accelerator.save_state(f'checkpoints/{model_name}_{epoch}_{i}', safe_serialization=False) # use model_name
            print(f"Saved model with loss {mean_loss}")

    pbar.close()

def train_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, accelerator, similarity_method, softmax_temperature=0.01, softargmax_beta=1000.0, model_name='best_model'):
    model.train()

    epoch_loss = []
    pbar = tqdm(total=len(dataloader), disable=(not accelerator.is_main_process))
    for i, batch in enumerate(dataloader):
        # Concatenate images
        images = torch.cat([batch['source_image'], batch['target_image']], dim=0)
        source_points = batch['source_points']
        target_points = batch['target_points']

        # Run through model
        features = model(images)

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

        # Calculate prediction and target
        if similarity_method == 'softmax': # cross-entropy
            kernel_size = 7 # kernel size for blurring target
            prediction = softmax_with_temperature(similarity_map, softmax_temperature) # [B, N, HxW]
            prediction = prediction.reshape(*prediction.shape[:2], h, w) # [B, N, H, W]
            target = one_hot(target_idxs, num_classes=similarity_map.shape[-1]).type(prediction.dtype) # [B, N, HxW]
            target = target.reshape(*target.shape[:2], h, w) # [B, N, H, W]
            target = torchvision.transforms.functional.gaussian_blur(target, kernel_size=kernel_size) # gaussian smooth target
        elif similarity_method == 'soft_argmax': # MSE
            epsilon = 1.0 # shift target points [epsilon, -epsilon] pixels
            similarity_map = similarity_map.reshape(*similarity_map.shape[:2], h, w) # [B, N, H, W]
            prediction = softargmax2d(similarity_map, softargmax_beta) # [B, N, 2]
            target = target_points + torch.randn_like(target_points) * epsilon # [B, N, 2]

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
        if i > 0 and i % int(len(dataloader) * 0.5) == 0: # every 50% of epoch
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            accelerator.save_state(f'checkpoints/{model_name}_{epoch}_{i}', safe_serialization=False) # use model_name
            print(f"Saved model with loss {mean_loss}")

    pbar.close()


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
    parser.add_argument('--cache_dir', type=str, default='cache', help='Path to cache directory')
    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument('--reset_cache', action='store_true', help='Reset cache')
    parser.add_argument('--only_tvmonitor', action='store_true', help='Only use tvmonitor category')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    model_config = args.model_config
    dataset_config = args.dataset_config
    num_workers = args.num_workers
    checkpoint = args.checkpoint
    cache_dir = args.cache_dir
    use_cache = args.use_cache
    reset_cache = args.reset_cache
    only_tvmonitor = args.only_tvmonitor

    # Load model config
    model_config = read_model_config(model_config)[model_name]

    # Get model parameters
    image_size = model_config.get('image_size', (512, 512))
    batch_size = model_config.get('batch_size', 8)
    num_epochs = model_config.get('num_epochs', 100)
    learning_rate = model_config.get('learning_rate', 1e-4)
    mode = model_config.get('mode', 'train')
    sampling_method = model_config.get('sampling_method', 'full')
    similarity_method = model_config.get('similarity_method', 'softmax')
    loss_function = model_config.get('loss_function', 'cross_entropy')
    scheduler_type = model_config.get('scheduler_type', 'constant')
    half_precision = model_config.get('half_precision', False)
    weight_decay = model_config.get('weight_decay', 0.0)
    softmax_temperature = model_config.get('softmax_temperature', 0.01)
    softargmax_beta = model_config.get('softargmax_beta', 1000.0)
    step_percent = model_config.get('step_percent', 0.5)

    # Load model(s)
    if mode == 'distill':
        teacher = load_model(model_config['teacher_name'], model_config['teacher_config'])
        student = load_model(model_config['student_name'], model_config['student_config'])
    elif mode == 'train':
        student = load_model(model_name, model_config)

    # Set parameters of teacher to not require gradients
    if mode == 'distill':
        for param in teacher.model1.extractor.parameters():
            param.requires_grad = False
        for param in teacher.model2.extractor.parameters():
            param.requires_grad = False

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)

    # Load dataset parameters
    config = dataset_config[dataset_name]
    config['split'] = 'train'
    random_sampling = config.get('random_sampling', False)
    normalize_image = config.get('normalize_image', False)

    preprocess = Preprocessor(image_size, rescale_data=True, image_range=[0, 1], normalize_image=normalize_image)
    dataset = load_dataset(dataset_name, config, preprocess)
    dataset.image_pair = True

    # Initialize accelerator
    student_config = model_config if mode == 'train' else model_config['student_config']
    full_fine_tune = student_config.get('linear_head', False) is False and student_config.get('rank', None) is None
    accelerator = Accelerator(log_with="tensorboard", project_config=ProjectConfiguration(
            project_dir=".",
            logging_dir="logs"
        ),
        mixed_precision="fp16" if half_precision else "no", # bf16 for A100
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=full_fine_tune)] # True for full fine-tune
    )

    # Cache dataset
    with accelerator.main_process_first():
        if use_cache:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            cache_path = os.path.join(cache_dir, f"{model_name}_{dataset_name}.h5")
            dataset = cache_dataset(teacher, dataset, cache_path, reset_cache, batch_size, num_workers, accelerator.device, half_precision)
            dataset.load_images = True
            teacher.to("cpu") # unload teacher from GPU

    # Filter for tv-monitor
    if only_tvmonitor:
        dataset.data = [sample for sample in dataset.data if sample['source_category'] == 'tvmonitor']

    # Create dataloader
    def get_collate_fn(batch_size):
        keys_to_stack = ['source_image', 'target_image', 'source_features', 'target_features']
        if batch_size == 1:
            keys_to_stack += ['source_points', 'target_points']

        def collate_fn(batch):
            output = {}
            for key in batch[0].keys():
                output[key] = [sample[key] for sample in batch]
                if key in keys_to_stack:
                    output[key] = torch.stack(output[key])
            return output
        
        return collate_fn
    
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=random_sampling,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 collate_fn=get_collate_fn(batch_size))
    
    # Create criterion, optimizer and scheduler
    if loss_function == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_function == 'sce':
        criterion = SCELoss(alpha=1.0, beta=1.0)

    optimizer = torch.optim.AdamW(student.params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

    if scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloader)*step_percent, gamma=0.1)
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")

    tracker_config = {
        "dataset_name": dataset_name,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "image_size": str(image_size),
        "mode": mode,
        "sampling_method": sampling_method,
        "similarity_method": similarity_method,
        "loss_function": loss_function,
        "scheduler_type": scheduler_type,
    }
    if mode == 'distill':
        tracker_config["teacher_name"] = model_config['teacher_name']
        tracker_config["student_name"] = model_config['student_name']
    elif mode == 'train':
        tracker_config["model_name"] = model_name

    if use_cache or mode == 'train':
        student, optimizer, scheduler, dataloader = accelerator.prepare(student, optimizer, scheduler, dataloader)
    else:
        teacher, student, optimizer, scheduler, dataloader = accelerator.prepare(teacher, student, optimizer, scheduler, dataloader)
    
    accelerator.register_for_checkpointing(student)
    accelerator.init_trackers(model_name, config=tracker_config)
    if checkpoint is not None:
        accelerator.load_state(checkpoint)

    # Print seperator
    if accelerator.is_main_process:
        print(f"\n{'='*30} Training {model_name} {'='*30}\n")
    
    signal.signal(signal.SIGTERM, signal_handler)

    # Run training
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        if mode == 'distill':
            distill_epoch(teacher, student, dataloader, criterion, optimizer, scheduler, epoch, accelerator, use_cache, sampling_method, softmax_temperature, softargmax_beta, model_name)
        elif mode == 'train':
            train_epoch(student, dataloader, criterion, optimizer, scheduler, epoch, accelerator, similarity_method, softmax_temperature, softargmax_beta, model_name)

    # Log end of training
    accelerator.end_training()

    if accelerator.is_main_process:
        print(f"\n{'='*30} Finished {'='*30}\n")