import torch
from utils.utils_3d import save_obj, savemeshtes2
import os
import imageio
import cv2
from tqdm import tqdm
import numpy as np

def save_3d_shape(mesh_v_list, mesh_f_list, root, idx):
    n_mesh = len(mesh_f_list)
    mesh_dir = os.path.join(root, 'mesh_pred')
    os.makedirs(mesh_dir, exist_ok=True)
    for i_mesh in range(n_mesh):
        mesh_v = mesh_v_list[i_mesh]
        mesh_f = mesh_f_list[i_mesh]
        mesh_name = os.path.join(mesh_dir, '%07d_%02d.obj' % (idx, i_mesh))
        save_obj(mesh_v, mesh_f, mesh_name)

def normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample, normalized_scale=1.0):
    vertices = mesh_v.cuda()
    scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
    mesh_v1 = vertices / scale * normalized_scale
    mesh_f1 = mesh_f.cuda()
    points, _ = kal.ops.mesh.sample_points(mesh_v1.unsqueeze(dim=0), mesh_f1, n_sample)
    return points


def inference_and_save_geo(G_ema, run_dir, grid_z):
    ###############
    ###############
    ##############################
    # import ipdb
    # ipdb.set_trace()
    # run_dir = 'debug/show_car_mesh_update_t_50'
    import kaolin as kal
    with torch.no_grad():
        # G_ema.update_w_avg()
        # save_mesh_idx = 0

        # This codebase doesn't have style mixing :)
        use_style_mixing = False
        truncation_phi = 1.0###################
        mesh_dir = os.path.join(run_dir, 'gen_geo_for_eval_phi_%.2f' % (truncation_phi))
        surface_point_dir = os.path.join(run_dir, 'gen_geo_surface_points_for_eval_phi_%.2f' % (truncation_phi))
        if use_style_mixing:
            mesh_dir += '_style_mixing'
        mesh_dir += '_128res'###############
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(surface_point_dir, exist_ok=True)
        n_gen = 1500 * 5  # ################## Let's generate as much as possible
        i_mesh = 0
        ###################
        # n_gen = 10
        # import ipdb
        # ipdb.set_trace()
        for i_gen in tqdm(range(n_gen)):
            gen_z = torch.randn_like(grid_z[0])
            # import ipdb
            # ipdb.set_trace()
            generated_mesh = G_ema.generate_3d_mesh(gen_z=gen_z, truncation_psi=truncation_phi)
            for mesh_v, mesh_f in zip(*generated_mesh):
                if mesh_v.shape[0] == 0:
                    continue
                save_obj(mesh_v.data.cpu().numpy(), mesh_f.data.cpu().numpy(),
                         os.path.join(mesh_dir, '%07d.obj' % (i_mesh)))
                # import ipdb
                # ipdb.set_trace()
                points = normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample=2048, normalized_scale=1.0)
                np.savez(os.path.join(surface_point_dir, '%07d.npz' % (i_mesh)), pcd=points.data.cpu().numpy())
                i_mesh += 1
            if i_mesh >= n_gen:
                break