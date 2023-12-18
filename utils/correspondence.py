import torch
import torchvision
from PIL import Image

def rescale_points(points, old_size, new_size):
    x_scale = new_size[0] / old_size[0]
    y_scale = new_size[1] / old_size[1]
    points = torch.multiply(points, torch.tensor([x_scale, y_scale]))
    return points

def rescale_bbox(bbox, old_size, new_size):
    x_scale = new_size[0] / old_size[0]
    y_scale = new_size[1] / old_size[1]
    bbox = torch.multiply(bbox, torch.tensor([x_scale, y_scale, x_scale, y_scale]))
    return bbox

def preprocess_image(image_pil, size):
    image_pil = image_pil.convert('RGB').resize(size, Image.BILINEAR)
    image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
    image = 2 * image - 1 # Normalize to [-1, 1]
    return image

def preprocess_points(points, old_size, new_size):
    points = rescale_points(points, old_size, new_size) # [N, 2]
    points = points[:, [1, 0]] # flip x and y axis to match image
    return points

def postprocess_points(points, old_size, new_size):
    points = points[:, [1, 0]] # flip x and y axis to match image
    points = rescale_points(points, old_size, new_size) # [N, 2]
    return points

def preprocess_bbox(bbox, old_size, new_size):
    bbox = rescale_bbox(bbox, old_size, new_size) # [4]
    bbox = bbox[[1, 0, 3, 2]] # flip x and y axis to match image
    return bbox

def points_to_idxs(points, size):
    # [B, N, 2]
    points_y = points[:, :, 0].clamp(0, size[1]-1)
    points_x = points[:, :, 1].clamp(0, size[0]-1)
    idxs = size[1] * torch.round(points_y) + torch.round(points_x)
    return idxs

def flatten_features(features):
    # [B, C, H, W] -> [B, HxW, C]
    return features.flatten(2).permute(0, 2, 1)

def normalize_features(features):
    # [B, W*H, C]
    return features / torch.linalg.norm(features, dim=-1).unsqueeze(-1)

def compute_correspondence(source_features, target_features, source_points, size):
    # Resize features to match scale of points
    source_features = torch.nn.functional.interpolate(source_features, size, mode="bilinear")
    target_features = torch.nn.functional.interpolate(target_features, size, mode="bilinear")

    # Use source points to get features
    source_idxs = points_to_idxs(source_points, size).long() # [B, N]
    source_features = flatten_features(source_features) # [B, HxW, C]
    source_features = normalize_features(source_features) # [B, HxW, C]
    source_features = source_features[torch.arange(source_features.shape[0])[:, None], source_idxs] # [B, N, C]
    
    # Calculate similarity map
    target_features = flatten_features(target_features) # [B, HxW, C]
    target_features = normalize_features(target_features) # [B, HxW, C]
    similarity_map = source_features @ target_features.transpose(1, 2) # [B, N, HxW]

    # Get max similarity for each point and convert to coordinates (y, x)
    predicted_idx = torch.argmax(similarity_map, dim=2) # [B, N]
    predicted_points = torch.stack([predicted_idx // size[1], predicted_idx % size[1]], dim=2) # [B, N, 2]
    return predicted_points

def compute_pck(predicted_points, target_points, size, threshold=0.1, target_bbox=None):
    distances = torch.linalg.norm(predicted_points - target_points, axis=-1)
    if target_bbox is None:
        pck = distances <= threshold * max(size)
    else:
        y1, x1, y2, x2 = target_bbox
        pck = distances <= threshold * max(x2 - x1, y2 - y1)
    return pck.sum().item()