# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

import math

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    # @property
    # def label_dim(self):
    #     assert len(self.label_shape) == 1
    #     return self.label_shape[0]

    # @property
    # def has_labels(self):
    #     return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

################################
class ImageFolderDatasetMe(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        data_camera_mode = 'shapenet_car',
        split = 'train',
                 use_white_back=False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.img_size = resolution
        self.use_white_back = use_white_back
        if data_camera_mode == 'shapenet_car':
            root = '/raid/raid/shapenet_render_res_1024/img/02958343'
            self.camera_root = '/raid/camera'
        elif data_camera_mode == 'shapenet_chair':
            root = '/raid/raid/shapenet_render_res_1024/img/03001627'
            self.camera_root = '/raid/camera'
        elif data_camera_mode == 'renderpeople':
            root = '/raid/renderPeople_obj_texture'
            self.camera_root = '/raid/camera'
        elif data_camera_mode == 'shapenet_motorbike':
            root = '/raid/shapenet_render_res_1024_tmp/img/03790512'
            self.camera_root = '/raid/shapenet_render_res_1024_tmp/camera'
        elif data_camera_mode == 'ts_house':
            root = '/raid/ts_house_render/img/House'
            self.camera_root = '/raid/ts_house_render/camera'########################################################################
        elif data_camera_mode == 'ts_animal':
            root = '/raid/ts_animal_render_1024/img/Mammal-442'
            self.camera_root = '/raid/ts_animal_render_1024/camera'
        else:
            raise  ValueError
        print('==> use shapenet data root: %s' % (root))

        if not os.path.exists(root):
            print('==> ERROR!!!! THIS SHOULD ONLY HAPPEN WHEN USING INFERENCE')
            root = path


        folder_list = sorted(os.listdir(root))

        if data_camera_mode == 'shapenet_chair' or data_camera_mode == 'shapenet_car' \
                or data_camera_mode == 'ts_animal' or data_camera_mode == 'shapenet_motorbike':
            # split = 'train'
            self.random_elevation_max = 30
            if data_camera_mode == 'shapenet_car':
                split_name = 'data/shapenet_car/%s.txt' % (split)
                if split == 'all':
                    split_name = 'data/shapenet_car.txt'
            elif data_camera_mode == 'shapenet_chair':
                split_name = 'data/shapenet_chair/%s.txt' % (split)
                if split == 'all':
                    split_name = 'data/shapenet_chair.txt'
            elif data_camera_mode == 'ts_animal':
                split_name = '/home/jungao/shared-def-tet/dataset/3dgan_data_split/ts_animals/%s.txt' % (split)
                # split_name = 'data/shapenet_car/%s.txt' % (split)
                if split == 'all':
                    raise NotImplementedError
                    # split_name = 'data/shapenet_car.txt'
            elif data_camera_mode == 'shapenet_motorbike':
                split_name = '/home/jungao/shared-def-tet/dataset/3dgan_data_split/shapenet_motorbike/%s.txt' % (split)
                # split_name = 'data/shapenet_chair/%s.txt' % (split)
                if split == 'all':
                    raise NotImplementedError
            else:
                #########
                raise NotImplementedError
                    # split_name = 'data/shapenet_chair.txt'
            valid_folder_list = []
            with open(split_name, 'r') as f:
                all_line = f.readlines()
                for l in all_line:
                    valid_folder_list.append(l.strip())
            valid_folder_list = set(valid_folder_list)
            useful_folder_list = set(folder_list).intersection(valid_folder_list)
            folder_list = sorted(list(useful_folder_list))
        if data_camera_mode == 'renderpeople':
            self.random_elevation_max = 20
        print('==> use shapenet folder number %s' % (len(folder_list)))
        folder_list = [os.path.join(root, f) for f in folder_list]
        all_img_list = []
        all_mask_list = []

        for folder in folder_list:
            rgb_list = sorted(os.listdir(folder))
            rgb_file_name_list = [os.path.join(folder, n) for n in rgb_list]
            all_img_list.extend(rgb_file_name_list)
            all_mask_list.extend(rgb_list)
        self.img_list = all_img_list
        self.mask_list = all_mask_list
        self._type = 'dir'#######
        PIL.Image.init()
        self._image_fnames = self.img_list
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        self._all_fnames = self._image_fnames
        name = os.path.splitext(os.path.basename(self._path))[0]

        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        self.label_dim = 25
        self.has_labels = True
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        image, condition = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy(), condition

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        ori_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        img = ori_img[:, :, :3][..., ::-1]
        mask = ori_img[:, :, 3:4]
        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)  ########

        img = resize_img.transpose(2, 0, 1)
        if self.use_white_back:
            white_background = np.zeros_like(img)
            # if self.white_background:
                # if self.data_camera_mode != 'stylegan_car':
            white_background += 255  ############## Just to make stylegan_car is the same for comparison
            img = img * (mask > 0).astype(np.float) + white_background * (1 - (mask > 0).astype(np.float))
            img = img.astype(np.uint8)
        condition = self.get_label(raw_idx)

        return img, condition

    def get_label(self, idx, use_random_label=False, random_elevation_max=30):
        fname = self._image_fnames[idx]
        fname_list = fname.split('/')
        img_idx = int(fname_list[-1].split('.')[0])
        obj_idx = fname_list[-2]
        syn_idx = fname_list[-3]

        if not os.path.exists(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy')):
            print('==> not found camera root')
        else:
            rotation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy'))
            elevation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'elevation.npy'))
            # img_idx = fname.split()
            #############
            # Rotation angle is 0~360
            if use_random_label:
                rotation_camera = np.random.rand(rotation_camera.shape[0]) * 360  # 0 ~ 360
                elevation_camera = np.random.rand(elevation_camera.shape[0]) * self.random_elevation_max  # ~ 0~30
            rotation = (-rotation_camera[img_idx] - 90) / 180 * np.pi
            elevation = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
            radius = 1.2


        return create_condition_from_camera_angle(rotation, elevation, radius)


    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
#----------------------------------------------------------------------------


# def projection(x=0.1, n=1.0, f=50.0, near_plane=None):
#     if near_plane is None:
#         near_plane = n
#     return np.array([[n / x, 0, 0, 0],
#                      [0, n / -x, 0, 0],
#                      [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
#                      [0, 0, -1, 0]]).astype(np.float32)

def create_my_world2cam_matrix( forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam
#
# def create_cam2world_matrix( forward_vector, origin, device=None):
#     """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
#
#     forward_vector = normalize_vecs(forward_vector)
#     up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
#
#     left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
#
#     up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))
#
#     rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
#     rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)
#
#     translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
#     translation_matrix[:, :3, 3] = origin
#
#     cam2world = translation_matrix @ rotation_matrix
#
#     return cam2world
#
# def create_camera_from_angle(phi, theta, sample_r, device='cuda'):
#     phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)
#
#     camera_origin = torch.zeros((phi.shape[0], 3), device=device)
#     camera_origin[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(theta)
#     camera_origin[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(theta)
#     camera_origin[:, 1:2] = sample_r * torch.cos(phi)
#
#     forward_vector = normalize_vecs(camera_origin)
#
#     world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device=device)
#     cam2world_matrix = torch.inverse(world2cam_matrix) # We know this is correct
#     return cam2world_matrix, forward_vector, camera_origin, phi, theta


def create_camera_matrix(rotation_angle, elevation_angle, camera_radius):
    device = 'cpu'
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

def create_condition_from_camera_angle( rotation, elevation, radius):

    fovy = np.arctan(32 / 2 / 35) * 2
    fovyangle = fovy / np.pi * 180.0
    ######################
    # # fov = fovyangle
    # focal = np.tan(fovyangle / 180.0 * np.pi * 0.5)
    # proj_mtx = projection(x=focal, f=1000.0, n=1.0, near_plane=0.1)
    elevation = torch.zeros(1) + elevation
    rotation = torch.zeros(1) + rotation
    radius = torch.zeros(1) + radius
    world2cam, _, _, _, _ = create_camera_matrix(rotation_angle=rotation, elevation_angle=elevation, camera_radius=radius)
    intrinsics = convert_intrinsic_toeg3d(fovyangle_y = fovyangle, ratio = 1.0)
    cam2world_mat = conver_extrinsic_toeg3d(world2cam.data.cpu().numpy()[0])
    # camera = viewpoint_estimator
    condition = np.concatenate([cam2world_mat.reshape(-1), intrinsics.reshape(-1)]).astype(
        np.float32)
    return condition



######################

def perspectiveprojectionnp_wz(fovy, ratio=1.0, near=0.01, far=10.0):
    tanfov = np.tan(fovy / 2.0)
    mtx = [[1.0 / (ratio * tanfov), 0, 0, 0], \
           [0, 1.0 / tanfov, 0, 0], \
           [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], \
           [0, 0, -1.0, 0]]
    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=np.float32)


def convert_intrinsic_toeg3d(fovyangle_y, ratio):
    # ratio:  w/h
    fovy_y = fovyangle_y / 180.0 * np.pi
    mtx = perspectiveprojectionnp_wz(fovy_y, ratio)

    fx, fy = mtx[:2]
    fx = fx / 2
    fy = fy / 2
    cx = 0.5
    cy = 0.5

    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)

    return mtx


def conver_extrinsic_toeg3d(pos_world2cam_4x4):
    P_c2w_4x4 = np.linalg.inv(pos_world2cam_4x4)

    return P_c2w_4x4


#########################################################################
# wz's modification
#########################################################################
#
#
# #########################################################################
# # wz's modification
# #########################################################################
# def perspectiveprojectionnp_wz(fovy, ratio=1.0, near=0.01, far=10.0):
#     tanfov = np.tan(fovy / 2.0)
#     mtx = [[1.0 / (ratio * tanfov), 0, 0, 0], \
#            [0, 1.0 / tanfov, 0, 0], \
#            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], \
#            [0, 0, -1.0, 0]]
#     return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=np.float32)
#
#
# def convert_intrinsic_toeg3d(fovyangle_y, ratio):
#     # ratio:  w/h
#     mtx = perspectiveprojectionnp_wz(fovyangle_y, ratio)
#
#     fx, fy = mtx[:2]
#     fx = fx / 2
#     fy = fy / 2
#     cx = 0.5
#     cy = 0.5
#
#     mtx = np.array([[fx, 0, cx],
#                     [0, fy, cy],
#                     [0, 0, 1]], dtype=np.float32)
#
#     return mtx
#
#
# def conver_extrinsic_toeg3d(pos_world2cam_4x4):
#     # how to do converstion?
#     # 1) shape_world_ogl to shape_cam_ogl
#     # 2) shape_cam_ogl 2 shape_cam_opencv
#
#     # P_c = R * P_w + T
#     # P_c_opencv = R_inv * P_c
#     # P_c_opencv = R_inv * R * P_w + R_inv * T
#     # import ipdb
#     # ipdb.set_trace()
#     R_w2c_3x3 = pos_world2cam_4x4[:3, :3]
#     T_w2c_3x1 = pos_world2cam_4x4[:3, 3:4]
#
#     R_inv = np.eye(3)
#     # rotate around x by 180
#     R_inv[1, 1] = -1
#     R_inv[2, 2] = -1
#
#     R_w2c_opencv_3x3 = R_inv * R_w2c_3x3
#     T_w2c_opencv_3x1 = R_inv * T_w2c_3x1
#     R_w2c_opencv_3x3 = np.matmul(R_inv, R_w2c_3x3)
#     T_w2c_opencv_3x1 = np.matmul(R_inv, T_w2c_3x1)
#
#     P_w2c_4x4 = np.eye(4, dtype=np.float32)
#     P_w2c_4x4[:3, :3] = R_w2c_opencv_3x3
#     P_w2c_4x4[:3, 3:4] = T_w2c_opencv_3x1
#
#     P_c2w_4x4 = np.linalg.inv(P_w2c_4x4)
#
#     return P_c2w_4x4

#########################################################################
# wz's modification
#########################################################################
