import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import scipy.ndimage
# from sklearn.metrics import f1_score
# import pydicom as dicom
# import nrrd

import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
# from torchsummary import summary

from utils.shlynur_print_time import print_time
from utils.shlynur_create_brats_data import load_brats_data_3d
from utils.shlynur_load_prostate_data import load_prostate_images_and_labels_3d
from utils.shlynur_class_dice import class_dice_3d_prostate
from utils.older_utils_functions import class_dice_2d
from utils.shlynur_get_dataset import GetProstateDataset, GetBratsDataset
from utils.shlynur_unet_models import UNet3D, Reversible3D
# from utils.shlynur_loss_functions import cross_entropy3d  # , brats_dice_loss, brats_dice_loss_original_4
from utils.shlynur_convergence_check import convergence_check_auto_cnn, prediction_converter, \
    identify_incorrect_pixels, mask_creator, accuracy_check_auto_inter_cnn
from utils.shlynur_save_load_checkpoint import save_checkpoint, load_checkpoint

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

# This program changes the starting epoch for the models. Needed to update the starting epoch if the model does not
#   finish training before the queue submission ends.
# Set first the hyper-parameters of the model (lines 52-71).
# Then specify older starting epochs, the model name and its location (lines 91-97).

seed = 48
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# # # # # # # # # # # # # # # #
# Change these parameters!!!! #
# # # # # # # # # # # # # # # #
bool_prostate_data = True
# model_type = 'gustav'
num_filters_auto_cnn = 32
num_filters_inter_cnn = 32
num_epochs_auto_cnn = 500
num_epochs_inter_cnn = 50
batch_size = 1
learning_rate = 3e-4
bool_binary_testing = False
bool_loss_summation = False
if bool_binary_testing:
    num_classes = 2
else:
    num_classes = 3 if bool_prostate_data else 5
channels_auto = 1 if bool_prostate_data else 4
channels_inter = channels_auto + num_classes + 1
if bool_loss_summation:
    channels_inter += 2
padding = (1, 1, 1) if bool_prostate_data else (0, 0, 0)
max_pooling_d = 1 if bool_prostate_data else 2

# Create the default models.
cnn_auto = UNet3D(bool_prostate_data=bool_prostate_data, in_channels=channels_auto,
                  number_of_classes=num_classes, number_of_filters=num_filters_auto_cnn,
                  kernel_size=(3, 3, 3), padding=padding, max_pooling_d=max_pooling_d).cuda()
cnn_inter = UNet3D(bool_prostate_data=bool_prostate_data, in_channels=channels_inter,
                   number_of_classes=num_classes, number_of_filters=num_filters_inter_cnn,
                   kernel_size=(3, 3, 3), padding=padding, max_pooling_d=max_pooling_d).cuda()

# Create the default optimizers and learning rate schedulers.
optimizer_auto = Adam(params=cnn_auto.parameters(), lr=learning_rate)
scheduler_auto = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_auto, mode='min', factor=0.99,
                                                patience=100, verbose=True)
optimizer_inter = Adam(params=cnn_inter.parameters(), lr=learning_rate)
scheduler_inter = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_inter, mode='min', factor=0.99,
                                                 patience=10, verbose=True)

# Set the current (older) epoch numbers for each model.
current_epochs = (48, 48, 48, 48)
model_path_name = 'inter_cnn_3D_model_gustav_epochs_50_filters_32_lr_0.0003_dataug_True.pt'
operating_system = 'linux'
if operating_system == 'linux':
    base_path = '/scratch_net/giggo/shlynur/msc/polybox/M.Sc.2019/code/shlynur_unet_testing/3d_results/'
else:
    base_path = r'D:\ETH_Projects\polybox\M.Sc.2019\code\shlynur_unet_testing\3d_results'

# if False, will only load and display the current starting epoch.
# If True, will load and update the starting epoch parameter of the .pt file.
update_model = False

for i in range(1, len(current_epochs) + 1):
    model_path = os.path.join(
        base_path, 'prostate_split_{}_iterations_10_20_mode_3d_num_scribbles_10'.format(i), 'models')
    os.chdir(path=model_path)
    print("GUSTAV'S SPLIT #{} at {}".format(i, model_path))
    current_epoch = current_epochs[i-1] - 1

    cnn_inter, optimizer_inter, scheduler_inter, starting_epoch_inter_cnn, inter_class_scores = load_checkpoint(
        model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, file_folder=model_path,
        filename=model_path_name)

    print("Starting epoch: {}".format(starting_epoch_inter_cnn))
    print("Current scores: {}".format(inter_class_scores))
    print("Current epoch:  {}".format(current_epoch))

    if update_model:
        save_checkpoint(model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, epoch=current_epoch,
                        val_score=inter_class_scores, model_name=os.path.join(model_path, model_path_name))

        cnn_inter, optimizer_inter, scheduler_inter, starting_epoch_inter_cnn, inter_class_scores = load_checkpoint(
            model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, file_folder=model_path,
            filename=model_path_name)

        print("Still Starting epoch: {}".format(starting_epoch_inter_cnn))
        print("Still Current scores: {}".format(inter_class_scores))
        print("New Current epoch:  {}".format(current_epoch))
