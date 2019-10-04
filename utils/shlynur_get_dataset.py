import torch
from torch.utils.data import Dataset
from utils.shlynur_data_augmentation import min_max_normalization, aug_flip, aug_rotate, aug_zoom, \
    aug_elastic_deform, aug_intensity_shift  # , brueggger_augment_3d_image
import h5py
import numpy as np
import cv2
# import random


class GetProstateDataset(Dataset):
    # Initialize your data by adding the links to the data files.
    def __init__(self, x_data, y_data, dimensions, transform=False, rot_angle=5, max_crop=1.1, deform_mu=0,
                 deform_sigma=10, max_intensity_shift=0.1, bool_binary=False):
        self.x_data = x_data
        self.y_data = y_data
        self.dimensions = dimensions
        self.len = []

        if self.dimensions == 2:
            self.len = self.x_data.shape[2]
        elif self.dimensions == 3:
            self.len = len(self.x_data)

        self.transform = transform
        self.rot_angle = rot_angle
        self.max_crop = max_crop
        self.deform_mu = deform_mu
        self.deform_sigma = deform_sigma
        self.max_intensity_shift = max_intensity_shift

        self.bool_binary = bool_binary

    def __getitem__(self, index):
        image, seg = [], []
        axes_flip_h, axes_flip_v, axes_rot = [], [], []
        if self.dimensions == 2:
            image = self.x_data[:, :, index]
            seg = self.y_data[:, :, index]
            axes_flip_h, axes_flip_v = 1, 0
            axes_rot = (1, 0)
        elif self.dimensions == 3:
            image = self.x_data[index]
            seg = self.y_data[index]
            axes_flip_h, axes_flip_v = 2, 1
            axes_rot = (1, 2)

        if self.bool_binary:
            np.place(seg, seg == 2, 1)

        if self.transform:
            image, seg = aug_flip(image, seg, axis=axes_flip_h)  # horizontal flip.
            # image, seg = aug_flip(image, seg, axis=axes_flip_v)  # vertical flip.
            image, seg = aug_rotate(image, seg, angle=self.rot_angle, axes=axes_rot)
            image, seg = aug_zoom(image, seg, scale_factor=self.max_crop, dimensions=self.dimensions)
            image, seg = aug_elastic_deform(image, seg, mu=self.deform_mu, sigma=self.deform_sigma,
                                            interpolation_method=cv2.INTER_NEAREST)
            image = aug_intensity_shift(image, bool_prostate_data=True, max_intensity_shift=self.max_intensity_shift)

        image = min_max_normalization(image, method='z_score')

        image = torch.from_numpy(image)
        seg = torch.from_numpy(seg)

        return image, seg

    def __len__(self):
        return self.len


# class GetProstateDatasetBinary(Dataset):
#     # Initialize your data by adding the links to the data files.
#     def __init__(self, x_data, y_data, dimensions, transform=False, rot_angle=5, max_crop=1.1, deform_mu=0,
#                  deform_sigma=10, max_intensity_shift=0.1):
#         self.x_data = x_data
#         self.y_data = y_data
#         self.dimensions = dimensions
#         self.len = []
#
#         if self.dimensions == 2:
#             self.len = self.x_data.shape[2]
#         elif self.dimensions == 3:
#             self.len = len(self.x_data)
#
#         self.transform = transform
#         self.rot_angle = rot_angle
#         self.max_crop = max_crop
#         self.deform_mu = deform_mu
#         self.deform_sigma = deform_sigma
#         self.max_intensity_shift = max_intensity_shift
#
#     def __getitem__(self, index):
#         image, seg = [], []
#         axes_flip_h, axes_flip_v, axes_rot = [], [], []
#         if self.dimensions == 2:
#             image = self.x_data[:, :, index]
#             seg = self.y_data[:, :, index]
#             axes_flip_h, axes_flip_v = 1, 0
#             axes_rot = (1, 0)
#         elif self.dimensions == 3:
#             image = self.x_data[index]
#             seg = self.y_data[index]
#             axes_flip_h, axes_flip_v = 2, 1
#             axes_rot = (1, 2)
#
#         np.place(seg, seg == 2, 1)
#
#         if self.transform:
#             # image, seg = brueggger_augment_3d_image(
#             #     image=image, label=seg, nn_aug=True, axes_flip_h=axes_flip_h, axes_flip_v=axes_flip_v,
#             #     rot_degrees=axes_rot, scale_factor=self.max_crop)
#
#             image, seg = aug_flip(image, seg, axis=axes_flip_h)
#             # image, seg = aug_flip(image, seg, axis=axes_flip_v)
#             image, seg = aug_rotate(image, seg, angle=self.rot_angle, axes=axes_rot)  # 15
#             image, seg = aug_zoom(image, seg, scale_factor=self.max_crop, dimensions=self.dimensions)  # 1.3
#             image, seg = aug_elastic_deform(image, seg, mu=self.deform_mu, sigma=self.deform_sigma,
#                                             interpolation_method=cv2.INTER_NEAREST)
#             image = aug_intensity_shift(image, bool_prostate_data=True, max_intensity_shift=self.max_intensity_shift)
#
#         image = min_max_normalization(image, method='z_score')
#
#         image = torch.from_numpy(image)
#         seg = torch.from_numpy(seg)
#
#         # print("image: {} \tseg: {} \tLength: {}".format(image.shape, seg.shape, self.len))
#
#         return image, seg
#
#     def __len__(self):
#         return self.len


class GetBratsDataset(Dataset):
    def __init__(self, file_path, mode, group='test', transform=False, rot_angle=5, max_crop=1.1, deform_mu=0,
                 deform_sigma=10, max_intensity_shift=0.1):
        super(GetBratsDataset, self).__init__()
        self.file_path = file_path
        self.group = group
        self.mode = mode
        self.file = None

        self.transform = transform
        self.rot_angle = rot_angle
        self.max_crop = max_crop
        self.deform_mu = deform_mu
        self.deform_sigma = deform_sigma
        self.max_intensity_shift = max_intensity_shift

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        if self.group == 'test':
            image = self.file[self.group + '/images_' + self.group][index, ...]
            seg = self.file[self.group + '/masks_' + self.group][index, ...]
        else:
            image = self.file[self.group + '/images_' + self.group + '_' + self.mode][index, ...]
            seg = self.file[self.group + '/masks_' + self.group + '_' + self.mode][index, ...]

        if self.transform:
            image, seg = aug_flip(image, seg, axis=2, bool_prostate_data=False)  # horizontal flip.
            # image, seg = aug_flip(image, seg, axis=1)  # vertical flip.
            image, seg = aug_rotate(image, seg, angle=self.rot_angle, axes=(1, 2), bool_prostate_data=False)
            image, seg = aug_zoom(image, seg, scale_factor=self.max_crop, dimensions=3, bool_prostate_data=False)
            image, seg = aug_elastic_deform(image, seg, mu=self.deform_mu, sigma=self.deform_sigma,
                                            interpolation_method=cv2.INTER_NEAREST, bool_prostate_data=False)
            image = aug_intensity_shift(image, max_intensity_shift=self.max_intensity_shift, bool_prostate_data=False)

        # Normalization
        new_image = []
        for i in range(image.shape[0]):
            new_image.append(min_max_normalization(image[i, :, :, :], method='z_score'))
        new_image = np.array(new_image)
        image = new_image

        image, seg = torch.from_numpy(image), torch.from_numpy(seg)
        return image, seg

    def __len__(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        if self.group == 'test':
            return self.file[self.group + '/images_' + self.group].shape[0]
        else:
            return self.file[self.group + '/images_' + self.group + '_' + self.mode].shape[0]
