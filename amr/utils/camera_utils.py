import math
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Union
from jaxtyping import Float
import numpy as np


def get_c2w(
        azimuth_deg,
        elevation_deg,
        camera_distances,):
    assert len(azimuth_deg) == len(elevation_deg) == len(camera_distances)
    n_views = len(azimuth_deg)
    #camera_distances = torch.full_like(elevation_deg, dis)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def build_normalization_matrix(vertices: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Compute the normalization matrix that maps original mesh coordinates to normalized coordinates.
    Normalization is defined as: v_norm = (v - center) * scale, mapping the longest bbox side to 2.0.
    Returns (N, scale, center) where N is a 4x4 matrix.
    """
    bounding_box_max = vertices.max(0)[0]
    bounding_box_min = vertices.min(0)[0]
    mesh_scale = 2.0
    scale = float(mesh_scale / ((bounding_box_max - bounding_box_min).max() + 1e-6))
    center_offset = (bounding_box_max + bounding_box_min) * 0.5

    N = torch.eye(4, dtype=torch.float32, device=vertices.device)
    N[0, 0] = scale
    N[1, 1] = scale
    N[2, 2] = scale
    N[:3, 3] = -center_offset * scale
    return N, scale, center_offset


def build_c2w(vertices: torch.Tensor, num_views: int = 1) -> Dict:
    """
    For sup views: Random elevation and azimuth, fixed distance and close fov.
    :param num_views: number of supervision views
    :param kwargs: additional arguments
    """
    # Default camera intrinsics
    default_elevation = 10
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
            math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, num_views + 1)[:num_views] - 90

    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)

    # Get the corresponding azimuth and elevation
    view_idxs = torch.arange(0, num_views)
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)
    fovy = torch.full_like(azimuth, default_fovy)
    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_fovy': fovy,
    }


def _get_projection_matrix(
    fovy: Union[float, Float[torch.Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[torch.Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def _get_mvp_matrix(
    c2w: Float[torch.Tensor, "*B 4 4"], proj_mtx: Float[torch.Tensor, "*B 4 4"]
) -> Float[torch.Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[torch.Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[torch.Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx