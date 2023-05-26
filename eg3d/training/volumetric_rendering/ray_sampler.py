"""
The ray sampler is a module that takes in camera matrices and returns ray bundles.
Lighter than the original ray sampler because it doesn't produce depths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import math_utils

class RaySampler(nn.Module):
    def __init__(self, camera_look_at_negative_z = False):
        super().__init__()
        self.camera_look_at_negative_z = camera_look_at_negative_z
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None

    def get_camera_params(self, uv, pose, intrinsics):
        cam_loc = pose[:, :3, 3]
        p = pose

        batch_size, num_samples, _ = uv.shape

        depth = torch.ones((batch_size, num_samples), device=uv.device)
        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        # import ipdb
        # ipdb.set_trace()
        if True or self.camera_look_at_negative_z:
            depth = -1 * depth
        z_cam = depth.view(batch_size, -1)# TODO, we have this only for our code

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

        world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_coords - cam_loc[:, None, :]
        ray_dirs = F.normalize(ray_dirs, dim=2)

        return ray_dirs, cam_loc

    def forward(self, cam2world_matrix, intrinsics, resolution):
        # import ipdb
        # ipdb.set_trace()
        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), indexing='ij')) * (1./resolution) + (0.5/resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        pose = cam2world_matrix
        ################
        # import ipdb
        # ipdb.set_trace()
        # intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * 0.5
        # intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * 0.5
        ray_dirs, cam_locs = self.get_camera_params(uv, pose, intrinsics)
        # import ipdb
        # ipdb.set_trace()
        # intrinsics[:, 0,0] = intrinsics[:, 0, 0] * 0.5
        # intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * 0.5
        # t = ray_dirs.reshape(-1, 64, 64, 3)
        # tt = t[:, 32, 32, :]
        # import ipdb
        # ipdb.set_trace()
        ######################################
        cam_locs = cam_locs.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return cam_locs, ray_dirs

def lift(x, y, z, intrinsics):
    # parse intrinsics
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z
    # import ipdb
    # ipdb.set_trace()
    # homogeneous
    # import ipdb
    # ipdb.set_trace()
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)