import torch

def flatten_features(features):
    # (b, c, w, h) -> (b, w*h, c)
    b, c, w, h = features.shape
    features = features.view((b, c, -1))
    features = features.permute((0, 2, 1))
    return features

def normalize_features(features):
    # (b, w*h, c)
    features = features / torch.linalg.norm(features, dim=-1)[:, :, None]
    return features

def cosine_similarity(source_features, target_features, flatten=True, normalize=True):
    if flatten:
        source_features = flatten_features(source_features)
        target_features = flatten_features(target_features)
    if normalize:
        source_features = normalize_features(source_features)
        target_features = normalize_features(target_features)
    sim = torch.matmul(source_features, target_features.permute((0, 2, 1)))
    return sim

def get_correspondences(source_features, target_features, source_points, output_size, load_size):
    """
    Precompute nearest neighbor of source_points in img1 to target_points in img2.
    """
    img1_feats = torch.nn.functional.interpolate(img1_feats, load_size, mode="bilinear")
    img2_feats = torch.nn.functional.interpolate(img2_feats, load_size, mode="bilinear")

    source_idx = torch.from_numpy(points_to_idxs(source_points, load_size)).long()
    # Select source_points in the flattened (w, h) dimension as source_idx
    img1_feats = flatten_feats(img1_feats)
    img2_feats = flatten_feats(img2_feats)
    img1_feats = img1_feats[:, source_idx, :]
    img1_feats = normalize_feats(img1_feats)
    img2_feats = normalize_feats(img2_feats)
    sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))

    # Find nn_correspondences but with points1 = source_points
    num_pixels = int(math.sqrt(sims.shape[-1]))
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % num_pixels
    points2_y = points2 // num_pixels
    points2 = torch.stack([points2_y, points2_x], dim=-1)

    points1 = torch.from_numpy(source_points)
    points2 = points2[0]
    return points1, points2