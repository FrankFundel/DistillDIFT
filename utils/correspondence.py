import torch

def flatten_features(features):
    # (b, c, w, h) -> (b, w*h, c)
    b, c, w, h = features.shape
    features = features.view((b, c, -1))
    features = features.permute((0, 2, 1))
    return features

def normalize_features(features):
    # (b, w*h, c)
    features = features / torch.linalg.norm(features, dim=-1).unsqueeze(-1)
    return features

def points_to_idxs(points, load_size):
    points_y = points[:, 0]
    points_y = torch.clip(points_y, 0, load_size[1]-1)
    points_x = points[:, 1]
    points_x = torch.clip(points_x, 0, load_size[0]-1)
    idx = load_size[1] * torch.round(points_y) + torch.round(points_x)
    return idx

def cosine_similarity(source_features, target_features, flatten=True, normalize=True):
    if flatten:
        source_features = flatten_features(source_features)
        target_features = flatten_features(target_features)
    if normalize:
        source_features = normalize_features(source_features)
        target_features = normalize_features(target_features)
    similarity = torch.matmul(source_features, target_features.permute((0, 2, 1)))
    return similarity

def compute_pck(predicted_points, target_points, load_size, pck_threshold=0.1, target_bounding_box=None):
    distances = torch.linalg.norm(predicted_points - target_points, axis=-1)
    if target_bounding_box is None:
        pck = distances <= pck_threshold * max(load_size)
    else:
        left, top, right, bottom = target_bounding_box
        pck = distances <= pck_threshold * max(right-left, bottom-top)
    return pck.sum(), pck.sum() / len(pck)


def get_correspondences(source_features, target_features, source_points, original_size):
    source_features = torch.nn.functional.interpolate(source_features, original_size, mode="bilinear")
    target_features = torch.nn.functional.interpolate(target_features, original_size, mode="bilinear")

    source_idx = points_to_idxs(source_points, original_size).long()
    # Select source_points in the flattened (w, h) dimension as source_idx
    source_features = flatten_features(source_features)
    target_features = flatten_features(target_features)
    source_features = source_features[:, source_idx, :]
    source_features = normalize_features(source_features)
    target_features = normalize_features(target_features)
    similarity = torch.matmul(source_features, target_features.permute((0, 2, 1)))

    # Find nn_correspondences but with points1 = source_points
    num_pixels = int(torch.sqrt(similarity.shape[-1]))
    predicted_points = similarity.argmax(dim=-1)
    predicted_points_x = predicted_points % num_pixels
    predicted_points_y = predicted_points // num_pixels
    predicted_points = torch.stack([predicted_points_x, predicted_points_y], dim=-1)

    predicted_points = predicted_points[0]
    return predicted_points