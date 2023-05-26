import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler_wz import RaySampler
import dnnlib
import numpy as np
from tqdm import tqdm
import kaolin as kal
from training.nvdiffrast_renderer import MyImportanceRenderer
########
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def gen_shape(shape_res=128, G=None, z=None,truncation_psi=1.0, truncation_cutoff=None, noise_mode=None, forward_label=None):
    max_batch = 1000000

    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                       cube_length=G.rendering_kwargs[
                                                                       'box_warp'] * 1)  # .reshape(1, -1, 3)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total=samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample(samples[:, head:head + max_batch],
                                 transformed_ray_directions_expanded[:, :samples.shape[1] - head], z, forward_label,
                                 truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                                 noise_mode=noise_mode)['sigma']
                sigmas[:, head:head + max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res))
    return sigmas


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of yzoutput color channels.
        sr_num_fp16_res     = 0,
                 camera_look_at_negative_z = False,
                 inference_show_geo_fid = False,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()

        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.inference_show_geo_fid = inference_show_geo_fid
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        # self.ray_sampler = RaySampler(camera_look_at_negative_z=camera_look_at_negative_z)
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.nvdiffrast_render = MyImportanceRenderer()
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def forward_instance(self, sample_coordinates, sample_directions):
        planes = self.backbone.synthesis(self.current_w, update_emas=False)

        # self.rendering_kwargs['depth_resolution'] = z_vals.shape[-1]//2
        # self.rendering_kwargs['depth_resolution_importance'] = z_vals.shape[-1]//2

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        out = self.renderer.run_model(planes, self.decoder, sample_coordinates.unsqueeze(0),
                                      sample_directions.unsqueeze(0), self.rendering_kwargs)
        return out

    def my_synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False,
                  generate_geo_img=False, rays=None, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # assert neural_rendering_resolution is not None
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # if rays is None:
            # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        # else:
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        # import pdb;pdb.set_trace()
        _, _, z_vals = rays

        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        self.rendering_kwargs['depth_resolution'] = z_vals.shape[-1]//2
        self.rendering_kwargs['depth_resolution_importance'] = z_vals.shape[-1]//2

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        feature_samples, depth_samples, weights_samples, all_depths, all_colors, all_densities \
            = self.renderer.my_forward(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs,
                                                                        generate_surface_only=False) # channels last
        #########################
        # RESHAPE INTO INPUT IMAGE
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        # sr_image = rgb_image = depth_image = None

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image,
                'all_depths': all_depths, 'all_colors': all_colors, 'all_densities': all_densities}

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False,
                  generate_geo_img=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # assert neural_rendering_resolution is not None
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs,
                                                                        generate_surface_only=False) # channels last
        #########################
        # RESHAPE INTO INPUT IMAGE
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}

    def my_generate_3d_mesh(self, gen_z, truncation_psi=0.7, use_w_space = False):
        max_batch = 1000000
        shape_res = 128 ##################### We only use 512 for visualization
        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                           cube_length=self.rendering_kwargs[
                                                                           'box_warp'] * 1)  # .reshape(1, -1, 3)

        samples = samples.to(gen_z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=gen_z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=gen_z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        c = torch.zeros(1, 25, device=gen_z.device)
        with torch.no_grad():
            while head < samples.shape[1]:
                sigma = self.my_sample(coordinates=samples[:, head:head + max_batch],
                                 directions=transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                    ws=gen_z, c=c,
                                 truncation_psi=truncation_psi)['sigma']
                sigmas[:, head:head + max_batch] = sigma
                head += max_batch


        sigmas = sigmas.reshape((shape_res, shape_res, shape_res))
        v_list, f_list = kal.ops.conversions.voxelgrids_to_trianglemeshes(sigmas.unsqueeze(dim=0), iso_value=20)
        normalized_v_list = []
        for v in v_list:
            new_v = v / shape_res - 0.5 # scale [-0.5, 0.5]
            new_v = new_v * self.rendering_kwargs['box_warp']
            normalized_v_list.append(new_v)

        return normalized_v_list, f_list


    def generate_3d_mesh(self, gen_z, truncation_psi=0.7, use_w_space = False):
        max_batch = 1000000
        shape_res = 128 ##################### We only use 512 for visualization
        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                           cube_length=self.rendering_kwargs[
                                                                           'box_warp'] * 1)  # .reshape(1, -1, 3)
        # import ipdb
        # ipdb.set_trace()
        samples = samples.to(gen_z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=gen_z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=gen_z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        c = torch.zeros(1, 25, device=gen_z.device)
        with torch.no_grad():
            while head < samples.shape[1]:
                sigma = self.sample(coordinates=samples[:, head:head + max_batch],
                                 directions=transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                    z=gen_z, c=c,
                                 truncation_psi=truncation_psi)['sigma']
                sigmas[:, head:head + max_batch] = sigma
                head += max_batch


        sigmas = sigmas.reshape((shape_res, shape_res, shape_res))
        v_list, f_list = kal.ops.conversions.voxelgrids_to_trianglemeshes(sigmas.unsqueeze(dim=0), iso_value=20)
        normalized_v_list = []
        for v in v_list:
            new_v = v / shape_res - 0.5 # scale [-0.5, 0.5]
            new_v = new_v * self.rendering_kwargs['box_warp']
            normalized_v_list.append(new_v)
        return normalized_v_list, f_list

    def generate_textured_mesh_img(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False,
                  generate_geo_img=False,mesh_v_list=None, mesh_f_list=None, **synthesis_kwargs):

        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        N = planes.shape[0]
        # feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins,
        #                                                                 ray_directions, self.rendering_kwargs,
        #                                                                 generate_surface_only=False)  # channels last

        feature_samples, depth_samples, weights_samples = self.nvdiffrast_render(planes, self.decoder, mesh_v_list,
                                                                        mesh_f_list, self.rendering_kwargs,
                                                                        generate_surface_only=False)  # channels last

        # RESHAPE INTO INPUT IMAGE
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})
        ####################
        mask_out_background = True
        if mask_out_background:
            weights_samples = weights_samples.permute(0, 3, 1, 2)
            sr_image = sr_image * weights_samples + (-1 * torch.ones_like(sr_image)) * (1 - weights_samples)
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}



    def my_sample(self, coordinates, directions, ws, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # ws = ws.repeat([ws.shape[0], self.backbone.mapping.num_ws, 1]) 
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False,
                cache_backbone=False, use_cached_backbone=False, generate_geo_img = False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        if self.inference_show_geo_fid:
            # import ipdb
            # ipdb.set_trace()
            ###############
            mesh_v_list, mesh_f_list = self.generate_3d_mesh(z, truncation_psi=truncation_psi)

            return self.generate_textured_mesh_img(ws, c, update_emas=update_emas,
                                                   neural_rendering_resolution=neural_rendering_resolution,
                                                   cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone,
                                                   mesh_v_list=mesh_v_list, mesh_f_list=mesh_f_list,
                                                   **synthesis_kwargs)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}
