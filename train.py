import torch
import signal
import argparse
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import torch.utils.data as data
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.nn.functional import interpolate

from utils.dataset import read_dataset_config, load_dataset, Preprocessor
from utils.model import read_model_config, load_model
from utils.correspondence import points_to_idxs, flatten_features, rescale_points

torch.backends.cuda.matmul.allow_tf32 = True

def distill(teacher, student, dataloader, criterion, optimizer, scheduler, num_epochs, strategy, accelerator):
    #teacher.eval()
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
                categories = batch['category'] * 2 #batch['source_category'] + batch['target_category']  #batch['category'] * 2

                # Run through model
                with torch.no_grad():
                    teacher_features = teacher(images, categories)
                student_features = student(images, categories)

                student_features = interpolate(student_features, teacher_features.shape[-2:], mode="bilinear") # this could eventually be replaced by a trainable upscaler
                #teacher_features = interpolate(teacher_features, student_features.shape[-2:], mode="bilinear") # maybe this is better/fairer

                # Normalize features
                student_source_features = torch.nn.functional.normalize(student_features[:len(student_features) // 2], dim=-1).flatten(2)
                student_target_features = torch.nn.functional.normalize(student_features[len(student_features) // 2:], dim=-1).flatten(2)
                teacher_source_features = torch.nn.functional.normalize(teacher_features[:len(teacher_features) // 2], dim=-1).flatten(2)
                teacher_target_features = torch.nn.functional.normalize(teacher_features[len(teacher_features) // 2:], dim=-1).flatten(2)

                # Calculate similarity
                if strategy == 'full':
                    student_similarity = student_source_features.transpose(-2, -1) @ student_target_features
                    teacher_similarity = teacher_source_features.transpose(-2, -1) @ teacher_target_features

                    visualize = True
                    if visualize:
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(2, len(student_source_features))
                        h, w = batch['source_image'].shape[2], batch['source_image'].shape[3]
                        for b in range(len(student_source_features)):
                            ax[0, b].imshow(batch['source_image'][b].permute(1, 2, 0).clamp(0, 1).cpu().detach())
                            ax[1, b].imshow(batch['target_image'][b].permute(1, 2, 0).clamp(0, 1).cpu().detach())
                            #source_points = torch.tensor(batch['source_points'][b]).unsqueeze(0)
                            #source_points = rescale_points(source_points.cpu(), (h, w), student_features.shape[-2:])
                            #rand_i = points_to_idxs(source_points, student_features.shape[-2:])[0, 0].item()
                            rand_i = 300 #torch.randint(0, student_source_features.shape[-1], (1,)).item()
                            source_with_point = torch.zeros(teacher_features.shape[-2:])
                            source_with_point[rand_i // student_features.shape[-2], rand_i % student_features.shape[-2]] = 1
                            ax[0, b].imshow(interpolate(source_with_point.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0), alpha=0.6)
                            ax[1, b].imshow(interpolate(student_similarity[b, rand_i, :].reshape(student_features.shape[-2:]).cpu().detach().unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0), alpha=0.6)
                            ax[0, b].axis('off')
                            ax[1, b].axis('off')
                        plt.savefig('student_similarity.png')

                        fig, ax = plt.subplots(2, len(teacher_source_features))
                        for b in range(len(teacher_source_features)):
                            ax[0, b].imshow(batch['source_image'][b].permute(1, 2, 0).clamp(0, 1).cpu().detach())
                            ax[1, b].imshow(batch['target_image'][b].permute(1, 2, 0).clamp(0, 1).cpu().detach())
                            #source_points = torch.tensor(batch['source_points'][b]).unsqueeze(0)
                            #source_points = rescale_points(source_points.cpu(), (h, w), student_features.shape[-2:])
                            #rand_i = points_to_idxs(source_points, student_features.shape[-2:])[0, 0].item()
                            rand_i = 300 #torch.randint(0, teacher_source_features.shape[-1], (1,)).item()
                            source_with_point = torch.zeros(teacher_features.shape[-2:])
                            source_with_point[rand_i // teacher_features.shape[-2], rand_i % teacher_features.shape[-2]] = 1
                            ax[0, b].imshow(interpolate(source_with_point.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0), alpha=0.6)
                            ax[1, b].imshow(interpolate(teacher_similarity[b, rand_i, :].reshape(teacher_features.shape[-2:]).cpu().detach().unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0), alpha=0.6)
                            ax[0, b].axis('off')
                            ax[1, b].axis('off')
                        plt.savefig('teacher_similarity.png')
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
                pbar.update(1) # TODO: maybe this should just be 1
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

def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, accelerator):
    pass

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
    strategy = model_config.get('strategy', 'full')

    # Load model
    teacher = load_model(model_config['teacher_name'], model_config['teacher_config'])
    student = load_model(model_config['student_name'], model_config['student_config'])

    # Load dataset config
    dataset_config = read_dataset_config(dataset_config)

    # Load dataset parameters
    config = dataset_config[dataset_name]
    random_sampling = config.get('random_sampling', False)

    #preprocess = Preprocessor(image_size, rescale_data=False)
    preprocess = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)), # ImageNet mean and std
    ])
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
    params = student.params_to_optimize if student.params_to_optimize else student.parameters()
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

    accelerator = Accelerator(log_with="tensorboard", project_config=ProjectConfiguration(
        project_dir=".",
        logging_dir="logs"
    ),
    mixed_precision="bf16",
    gradient_accumulation_steps=gradient_accumulation_steps)
    accelerator.init_trackers(model_name, config={
        "teacher_name": model_config['teacher_name'],
        "student_name": model_config['student_name'],
        "dataset_name": dataset_name,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "image_size": str(image_size),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        # loss function, method, scheduler etc.
    })
    teacher, student, optimizer, scheduler, dataloader = accelerator.prepare(teacher, student, optimizer, scheduler, dataloader)
    accelerator.register_for_checkpointing(student) #, optimizer, scheduler)
    if checkpoint is not None:
        accelerator.load_state(checkpoint)

    # Print seperator
    if accelerator.is_main_process:
        print(f"\n{'='*30} Training {model_name} {'='*30}\n")
    
    signal.signal(signal.SIGTERM, signal_handler)

    # Run training
    distill(teacher, student, dataloader, criterion, optimizer, scheduler, num_epochs, strategy, accelerator)

    if accelerator.is_main_process:
        print(f"\n{'='*30} Finished {'='*30}\n")