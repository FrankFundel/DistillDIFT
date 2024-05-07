import os
import torch
from typing import Optional, Tuple

from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.visualize import get_implicitron_sequence_pointcloud
from pytorch3d.implicitron.tools.point_cloud_utils import render_point_cloud_pytorch3d, _transform_points, PointsRasterizer, PointsRasterizationSettings


def get_depth_point_cloud_pytorch3d(
    camera,
    point_cloud,
    render_size: Tuple[int, int],
    point_radius: float = 0.03,
    topk: int = 10,
    eps: float = 1e-2,
    bin_size: Optional[int] = None,
    **kwargs,
):
    # move to the camera coordinates; using identity cameras in the renderer
    point_cloud = _transform_points(camera, point_cloud, eps, **kwargs)
    camera_trivial = camera.clone()
    camera_trivial.R[:] = torch.eye(3)
    camera_trivial.T *= 0.0

    bin_size = (
        bin_size
        if bin_size is not None
        else (64 if int(max(render_size)) > 1024 else None)
    )
    rasterizer = PointsRasterizer(
        cameras=camera_trivial,
        raster_settings=PointsRasterizationSettings(
            image_size=render_size,
            radius=point_radius,
            points_per_pixel=topk,
            bin_size=bin_size,
        ),
    )

    fragments = rasterizer(point_cloud, **kwargs)
    return fragments.zbuf.max(dim=-1)
