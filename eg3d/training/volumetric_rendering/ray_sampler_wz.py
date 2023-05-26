"""
The ray sampler is a module that takes in camera matrices and returns ray bundles.
Lighter than the original ray sampler because it doesn't produce depths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from . import math_utils

class RaySampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None

    def get_camera_params(self, uv_bxpx2, pose_bx4x4, intrinsics_bx3x3):

        r'''
        uv_bxpx2: 2D pixel coordinates, between (0, 1)
        pose_bx4x4: camera2world
        P_world = pose_bx4x4[:3, :3] * P_cam + pose_bx4x4[:3, 3:4]
        That's to say, camera origin (0, 0, 0) in camera space
        is actually pose_bx4x4[:3, 3:4] in world space
        '''

        cam_loc_world = pose_bx4x4[:, :3, 3]
        P_cam2world = pose_bx4x4

        batch_size, num_samples, _ = uv_bxpx2.shape

        depth = torch.ones((batch_size, num_samples), device=uv_bxpx2.device)
        x_cam = uv_bxpx2[:, :, 0].view(batch_size, -1)
        y_cam = uv_bxpx2[:, :, 1].view(batch_size, -1)
        z_cam = -depth.view(batch_size, -1)

        pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics_bx3x3)

        # permute for batch matrix product
        pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

        world_coords = torch.bmm(P_cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_coords - cam_loc_world[:, None, :]
        ray_dirs = F.normalize(ray_dirs, dim=2)

        return ray_dirs, cam_loc_world

    def forward(self, cam2world_matrix, intrinsics, resolution):

        # wz's modification
        # xycooridnate to
        if isinstance(resolution, int):
            resolution = [resolution, resolution]
        else:
            assert len(resolution) == 2
            assert isinstance(resolution, list) or isinstance(resolution, tuple)

        width, height = resolution

        xidx = torch.arange(width, dtype=torch.float32, device=cam2world_matrix.device)
        yidx = torch.arange(height, dtype=torch.float32, device=cam2world_matrix.device)

        # note that this y is not flip!!!
        xidx = (xidx + 0.5) / width
        yidx = (yidx + 0.5) / height
        yidx = 1 - yidx
        uv = torch.stack(torch.meshgrid(yidx, xidx))

        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        pose = cam2world_matrix


        ray_dirs, cam_locs = self.get_camera_params(uv, pose, intrinsics)

        cam_locs = cam_locs.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return cam_locs, ray_dirs

def lift(x, y, z, intrinsics):
    r'''
    z is negative
    '''
    # parse intrinsics
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    # x_pix: normaled pixel coordinate
    # x_plane: coordinate in image plane(z=1)
    # assume sk = 0
    # the formula is changed to:
    # x_pix = x_plane * fx + cx
    # y_pix = y_plane * fy + cy

    x_plane = (x - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
    y_plane = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1)

    # in opengl space, we should time with -z
    # in opencv space, we should time with z
    x_lift = x_plane * -z
    y_lift = y_plane * -z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)



if __name__ == '__main__':

    raysamobj = RaySampler( )

    bs = 5
    cam2world_matrix = torch.eye(4).reshape(-1, 4, 4).repeat(bs, 1, 1)
    intrinsics = torch.eye(3).reshape(-1, 3, 3).repeat(bs, 1, 1)
    resolution = [800, 600]
    re = raysamobj(cam2world_matrix,  intrinsics, resolution)


