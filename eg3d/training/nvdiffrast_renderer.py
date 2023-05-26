import cv2
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# import nvdiffrast.torch as dr
# import kaolin as kal

# def interpolate(attr, rast, attr_idx, rast_db=None):
#     return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
#                           diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def create_my_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat((left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam

def create_camera_matrix(rotation_angle, elevation_angle, camera_radius):
    device = 'cuda'
    n = rotation_angle.shape[0]
    phi = elevation_angle
    theta = rotation_angle
    sample_r = camera_radius
    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = sample_r * torch.cos(phi)
    camera_origin = output_points

    forward_vector = normalize_vecs(camera_origin)

    world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device=device)
    return world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle


class Camera(nn.Module):
    def __init__(self):
        super(Camera, self).__init__()
        pass


def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
    if near_plane is None:
        near_plane = n
    return np.array([[n / x, 0, 0, 0],
                     [0, n / -x, 0, 0],
                     [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
                     [0, 0, -1, 0]]).astype(np.float32)



class PerspectiveCamera(Camera):
    def __init__(self, fovy=49.0, device='cuda'):
        super(PerspectiveCamera, self).__init__()
        self.device = device
        focal = np.tan(fovy / 180.0 * np.pi * 0.5)
        self.proj_mtx = torch.from_numpy(projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)).to(self.device).unsqueeze(dim=0)

    def project(self, points_bxnx4):
        out = torch.matmul(points_bxnx4,
                           torch.transpose(self.proj_mtx, 1, 2))
        return out

# def create_condition_from_camera_angle( rotation, elevation, radius):
#
#     fovy = np.arctan(32 / 2 / 35) * 2
#     fovyangle = fovy / np.pi * 180.0
#     ######################
#     # # fov = fovyangle
#     # focal = np.tan(fovyangle / 180.0 * np.pi * 0.5)
#     # proj_mtx = projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)
#     elevation = torch.zeros(1) + elevation
#     rotation = torch.zeros(1) + rotation
#     radius = torch.zeros(1) + radius
#     world2cam, _, _, _, _ = create_camera_matrix(rotation_angle=rotation, elevation_angle=elevation, camera_radius=radius)
#     intrinsics = convert_intrinsic_toeg3d(fovyangle_y = fovyangle, ratio = 1.0)
#     cam2world_mat = conver_extrinsic_toeg3d(world2cam.data.cpu().numpy()[0])
#     # camera = viewpoint_estimator
#     condition = np.concatenate([cam2world_mat.reshape(-1), intrinsics.reshape(-1)]).astype(
#         np.float32)
#     return condition

class NvdiffrastRenderer(nn.Module):
    def __init__(self, device='cuda'):
        super(NvdiffrastRenderer, self).__init__()
        # Create fovy angle matrix
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        fov = fovyangle
        self.device = device
        self.camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        # self.ctx = dr.RasterizeGLContext(device=self.device)

    def sample_camera(self, random_elevation_max=30):
        rotation_camera = np.random.rand(2) * 360  # 0 ~ 360
        elevation_camera = np.random.rand(2) * random_elevation_max  # ~ 0~30

        rotation = (-rotation_camera - 90) / 180 * np.pi
        elevation = (90 - elevation_camera) / 180.0 * np.pi
        radius = 1.2
        rotation = torch.from_numpy(rotation).float().cuda()
        elevation = torch.from_numpy(elevation).float().cuda()
        camera_radius = torch.zeros_like(rotation) + radius
        # import ipdb
        # ipdb.set_trace()
        world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = \
            create_camera_matrix(camera_radius=camera_radius.unsqueeze(dim=-1), rotation_angle=rotation.unsqueeze(dim=-1),
                                 elevation_angle=elevation.unsqueeze(dim=-1))
        return world2cam_matrix[0:1]

    def vertices_to_face_v(self, vertices, face):
        v_a = vertices[face[:, 0], :].unsqueeze(dim=1)
        v_b = vertices[face[:, 1], :].unsqueeze(dim=1)
        v_c = vertices[face[:, 2], :].unsqueeze(dim=1)
        face_v = torch.cat([v_a, v_b, v_c], dim=1)
        return face_v


    def render_mesh_random_camera(self, mesh_v_list, mesh_f_list, resolution, random_elevation_max=30):
        # return_value = dict()
        all_3d_pos = []
        all_mask = []
        high_res_mask = []
        for mesh_v, mesh_f in zip(mesh_v_list, mesh_f_list):
            if mesh_v.shape[0] == 0:
                hard_mask = torch.zeros(1, resolution * 4, resolution * 4, 1, device=mesh_v.device)
                gb_feat = torch.zeros(1, resolution, resolution, 3, device=mesh_v.device)
                # hard_mask = high_res_render_feat[..., 3:4]
                # cv2.imwrite('debug/debug_kaolin_mask.png', hard_mask[0].data.cpu().numpy() * 255)
                # gb_feat = render_feat[..., :3]
                # hard_mask = torch.clamp(rast[..., -1:], 0, 1)
                all_mask.append(hard_mask)
                all_3d_pos.append(gb_feat)
                continue
            mesh_v_pos_bxnx3 = mesh_v.unsqueeze(dim=0)
            camera_mv_bx4x4 = self.sample_camera(random_elevation_max=random_elevation_max)
            mesh_v_feat_bxnxd = mesh_v_pos_bxnx3.detach().clone()
            mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=self.device) if not torch.is_tensor(
                camera_mv_bx4x4) else camera_mv_bx4x4
            v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
            v_pos_clip = self.camera.project(v_pos)  # Projection in the camera

            # Render the image,
            # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
            num_layers = 1
            # import ipdb
            # ipdb.set_trace()
            # import ipdb
            # ipdb.set_trace()
            face_vertices_image = self.vertices_to_face_v(v_pos_clip[0], mesh_f.long()).unsqueeze(dim=0)
            face_vertices_world = self.vertices_to_face_v(mesh_v, mesh_f.long()).unsqueeze(dim=0)
            face_vertices_image = face_vertices_image[..., :3] / face_vertices_image[..., 3:4]##########################
            face_vertices_image = torch.cat([
                face_vertices_image[..., 0:1],
             - face_vertices_image[..., 1:2],
               - face_vertices_image[..., 2:3]
            ], dim=-1)############## Flip y-direction###################
            face_vertices_feat = torch.cat([face_vertices_world, torch.ones_like(face_vertices_world[..., :1])], dim=-1)

            render_feat, _, _ = kal.render.mesh.dibr_rasterization(height=resolution, width=resolution,
                                                                   face_vertices_z=face_vertices_image[..., 2],
                                                                   face_vertices_image=face_vertices_image[..., :2],
                                                                   face_features=face_vertices_feat,
                                                                face_normals_z=torch.ones_like(face_vertices_image[..., :1]).squeeze(dim=-1), )
            # render_feat, _ = kal.render.mesh.rasterize(height=resolution, width=resolution,
            #                           face_vertices_z=torch.ones_like(face_vertices_image[..., :1]).squeeze(dim=-1),
            #                           face_vertices_image=face_vertices_image,
            #                           face_features=face_vertices_feat)
            ####################################################################################
            ################################################################################################################
            high_res_render_feat, _ = kal.render.mesh.rasterize(height=resolution * 4, width=resolution * 4,
                                                       face_vertices_z=face_vertices_image[..., 2],
                                                       face_vertices_image=face_vertices_image[..., :2],
                                                       face_features=face_vertices_feat)
            ####
            # with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_f,
            #                     [resolution, resolution]) as peeler:
            #     for _ in range(num_layers):
            #         rast, db = peeler.rasterize_next_layer()
            #         gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_f)
            ######################
            # import ipdb
            # ipdb.set_trace()
            hard_mask = high_res_render_feat[..., 3:4]
            # cv2.imwrite('debug/debug_kaolin_mask.png', hard_mask[0].data.cpu().numpy() * 255)
            gb_feat = render_feat[..., :3]
            # hard_mask = torch.clamp(rast[..., -1:], 0, 1)
            all_mask.append(hard_mask)
            all_3d_pos.append(gb_feat)
        return torch.cat(all_3d_pos, dim=0), torch.cat(all_mask, dim=0)




def generate_planes():
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    # Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    # import ipdb
    # ipdb.set_trace()
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    # don't worry, I've already made sure this does actually return identical output as F.grid_sample
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
def save_pointcloud(points, save_file_name, colors=None):
    if colors is None:
        colors = np.ones_like(points)
    d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2],
         'red': colors[:, 0], 'green': colors[:, 1], 'blue': colors[:, 2]}

    cloud = PyntCloud(pd.DataFrame(data=d))
    cloud.to_file(save_file_name)

class MyImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.renderer = NvdiffrastRenderer()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, mesh_v_list, mesh_f_list, rendering_options, generate_surface_only=False):
        ####################################
        # import ipdb
        # ipdb.set_trace()
        resolution = 128
        random_elevation_max = 45 ## This is only for animal !!!!!
        self.plane_axes = self.plane_axes.to(mesh_v_list[0].device)
        sample_coordinates, samples_mask = self.renderer.render_mesh_random_camera(mesh_v_list, mesh_f_list,
                                                                                   resolution=resolution, random_elevation_max=random_elevation_max)# The resolution is 128

        sample_coordinates = sample_coordinates.reshape(sample_coordinates.shape[0], -1, 3)
        sample_directions = torch.zeros_like(sample_coordinates)

        sample_directions[..., -1] = -1
        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse * 2 - 1  # Scale it to [-1, 1]
        return colors_coarse, torch.ones_like(colors_coarse[..., 0:1]), samples_mask
        rgb_final = colors_coarse.reshape(colors_coarse.shape[0], resolution, resolution, -1)
        depth_final = torch.ones_like(rgb_final[..., 0])
        weights = samples_mask
        # rgb_fina
        # rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options, generate_surface_only=generate_surface_only)
        ########

        return rgb_final, depth_final, weights

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])

        out = decoder(sampled_features, sample_directions)
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            depths_coarse = torch.linspace(ray_start,
                                    ray_end,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

