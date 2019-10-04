import argparse
import os
import time
import random
# from comet_ml import Experiment

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
# from utils.shlynur_load_brats_data import load_brats_data_3d
from utils.shlynur_load_prostate_data import load_prostate_images_and_labels_3d, load_prostate_images_and_labels_3d_test
from utils.shlynur_class_dice import class_dice_3d_prostate
from utils.older_utils_functions import class_hd_hd95
from utils.shlynur_get_dataset import GetProstateDataset, GetBratsDataset
from utils.shlynur_unet_models import UNet3D, Reversible3D
# from utils.shlynur_loss_functions import cross_entropy3d  # , brats_dice_loss, brats_dice_loss_original_4
from utils.shlynur_convergence_check import convergence_check_auto_cnn, prediction_converter, \
    prediction_converter_without_max, identify_incorrect_pixels, mask_creator, accuracy_check_auto_inter_cnn
from utils.shlynur_save_load_checkpoint import save_checkpoint, load_checkpoint
from run_gustavs_splits import print_memory_usage, print_current_epoch_results, plot_accuracy_images, \
    plot_inter_cnn_images, plot_loss_images

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def brats_prostate_gustavs_splits(
        seed=48, bool_prostate_data=True, model_type='gustav', num_filters_auto_cnn=32, num_filters_inter_cnn=32,
        num_epochs_auto_cnn=500, num_epochs_inter_cnn=50, batch_size=1, learning_rate=3e-4, lr_wd=1e-4,
        bool_load_previous_checkpoint=(False, False), load_model_folder=['', ''], load_model_file=['', ''],
        loss_weight=None, max_iterations_train=10, max_iterations_test=20, bool_auto_cnn_train=True,
        bool_inter_cnn_train=True, bool_save_results=True, bool_data_augmentation=True, inter_number_of_slices='all',
        inter_scribble_mode='2d', operating_system='linux', gustav_split=1, bool_binary_testing=False,
        bool_loss_summation=False, number_of_scribbles=1, size_of_scribble=4, bool_best_placement=False):
    """

    :param seed:
    :param bool_prostate_data:
    :param model_type:
    :param num_filters_auto_cnn:
    :param num_filters_inter_cnn:
    :param num_epochs_auto_cnn:
    :param num_epochs_inter_cnn:
    :param batch_size:
    :param learning_rate:
    :param lr_wd: weight decay of the learning rate
    :param bool_load_previous_checkpoint: A list containing True/False statements to load a previous checkpoint.
        If the first item is True, a previous autoCNN checkpoint is loaded.
        If the second item is True, a previous interCNN checkpoint is loaded.
    :param load_model_folder: The folder/s that contain the model/s to be loaded if the previous boolean is True.
    :param load_model_file: The model/s that will be loaded if the previous boolean is True.
    :param loss_weight:
    :param max_iterations_train:
    :param max_iterations_test:
    :param bool_auto_cnn_train:
    :param bool_inter_cnn_train:
    :param bool_save_results:
    :param bool_data_augmentation:
    :param inter_number_of_slices:
    :param inter_scribble_mode:
    :param operating_system:
    :param gustav_split:
    :param bool_binary_testing:
    :param bool_loss_summation:
    :param number_of_scribbles:
    :param size_of_scribble:
    :param bool_best_placement:

    :return:
    """
    start_timer = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bool_prostate_data:
        print('=== Every volume now has 15 slices. ===')
    print('*' * 25 + 'INTER SCRIBBLE MODE: {}'.format(inter_scribble_mode) + '*' * 25)
    if bool_binary_testing:
        print('Binary testing stuff')

    save_folder = '/scratch_net/giggo/shlynur/msc/polybox/M.Sc.2019/code/shlynur_unet_testing/3d_results' \
                  '/older_incorrect_stuff'
    save_folder = os.path.join(save_folder, '{}_using_gustavs_splits_{}_iterations_{}_{}_mode_{}'.format(
        'prostate' if bool_prostate_data else 'brats', gustav_split, max_iterations_train, max_iterations_test,
        inter_scribble_mode))
    if bool_loss_summation:
        save_folder += '_loss_sum'
    if bool_binary_testing:
        save_folder += '_binary'
    if number_of_scribbles != 1:
        save_folder += '_num_scribbles_{}'.format(number_of_scribbles)
    if size_of_scribble != 4:
        save_folder += '_size_scribble_{}'.format(size_of_scribble)
    if bool_best_placement:
        save_folder += '_lcc'
    if learning_rate != 3e-4:
        save_folder += '_lr_{}'.format(learning_rate)
    if lr_wd != 1e-4:
        save_folder += '_lrwd_{}'.format(lr_wd)

    print("#" * 29 + " USING GUSTAV'S SPLITS NO. {} ".format(gustav_split) + "#" * 29)

    if bool_binary_testing:
        num_classes = 2
    else:
        num_classes = 3 if bool_prostate_data else 5
    print("Results are in folder '{}'".format(save_folder))
    print("Seed = {} \t\t\tnum_classes = {} \nnum_filters_autoCNN = {} \tnum_filters_interCNN = {} \n"
          "num_epochs_autoCNN = {} \tnum_epochs_interCNN = {} \nbatch_size = {} \t\t\tlearning_rate = {}"
          "\nlr_weight_decay = {} \tweight = {} \nmax_iterations_train = {} \tmax_iterations_test = {} "
          "\nnumber_of_scribbles = {} \tsize_of_scribble = {} \nbool_best_placement = {}"
          "\nGustav split: {} \n"
          .format(seed, num_classes, num_filters_auto_cnn, num_filters_inter_cnn, num_epochs_auto_cnn,
                  num_epochs_inter_cnn, batch_size, learning_rate, lr_wd, loss_weight, max_iterations_train,
                  max_iterations_test, number_of_scribbles, size_of_scribble, bool_best_placement, gustav_split))
    print("Dataset: {}".format('NCI-ISBI Prostate' if bool_prostate_data else 'MICCAI BraTS'))
    print("Data augmentation: {}".format('True' if bool_data_augmentation else 'False'))
    print("Saving results: {}".format('True' if bool_save_results else 'False'))
    if bool_auto_cnn_train & bool_inter_cnn_train:
        training_str = "auto_CNN & inter_CNN."
    elif bool_auto_cnn_train:
        training_str = "auto_CNN."
    elif bool_inter_cnn_train:
        training_str = "inter_CNN."
    else:
        training_str = "None."
    print("Training: " + training_str)

    if loss_weight is not None:
        loss_weight = loss_weight.cuda()
    channels_auto = 1 if bool_prostate_data else 4
    channels_inter = channels_auto + num_classes + 1
    if bool_loss_summation:
        channels_inter += 2
    # channels_inter = 5 if bool_prostate_data else 10
    padding = (1, 1, 1) if bool_prostate_data else (0, 0, 0)
    max_pooling_d = 1 if bool_prostate_data else 2
    num_workers = 8 if bool_prostate_data else 8
    model_name_auto = "auto_cnn_3D_model_{}_epochs_{}_filters_{}_lr_{}_dataug_{}.pt".format(
        model_type, num_epochs_auto_cnn, num_filters_auto_cnn, learning_rate, bool_data_augmentation)
    model_name_inter = "inter_cnn_3D_model_{}_epochs_{}_filters_{}_lr_{}_dataug_{}.pt".format(
        model_type, num_epochs_inter_cnn, num_filters_inter_cnn, learning_rate, bool_data_augmentation)

    # Set the location to save everything
    os.makedirs(save_folder, exist_ok=True)
    os.chdir(save_folder)
    os.makedirs('models', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    os.makedirs('intercnn_images', exist_ok=True)
    os.makedirs('iterations', exist_ok=True)

    if bool_prostate_data:
        # Set the location of the data
        if operating_system == 'linux':
            data_folder = '/scratch_net/giggo/shlynur/msc/NCI_Prostate/'
            gustav_splits_folder = '/scratch_net/giggo/shlynur/msc/polybox/M.Sc.2019/code/shlynur_unet_testing/' \
                                   'gustav_code/Prostate_Data-splits'
        elif operating_system == 'windows':
            data_folder = r'D:\\ETH_Projects\\polybox\\M.Sc.2019\\NCI_Prostate'
            gustav_splits_folder = (r'D:\ETH_Projects\polybox\M.Sc.2019\code\shlynur_unet_testing'
                                    r'\gustav_code\Prostate_Data-splits')
        else:
            raise ValueError('The operating system must be either "linux" or "windows".')
        data_path_tr = os.path.join(data_folder, 'Prostate-3T/Prostate3T-01-')
        seg_path_tr = os.path.join(data_folder, 'NCI_ISBI_Challenge-Prostate3T_Training_Segmentations/Prostate3T-01-')

        # Create a random list
        # list_total = random.sample(range(1, 31), 30)
        # list_total.remove(19)
        # list_total = ['{:04d}'.format(x) for x in list_total]

        list_gustav_split = np.load(os.path.join(gustav_splits_folder,
                                                 'data_split_v{}.npy'.format(gustav_split)))
        list_total = ['{:04d}'.format(x) for x in list_gustav_split]

        # Split the list into training, testing and validating for different methods.
        list_auto_train = list_total[:15]   # [:15]
        list_auto_val = list_total[15:23]   # [15:23]
        # list_inter_train = list_total[:23]
        list_inter_val = [list_total[23]]   # [23]
        list_test = list_total[24:]         # [24:]

        # Load the images and masks using the previously created lists
        auto_cnn_img_train, auto_cnn_labels_train = load_prostate_images_and_labels_3d(list_auto_train, data_path_tr,
                                                                                       seg_path_tr)
        auto_cnn_img_val, auto_cnn_labels_val = load_prostate_images_and_labels_3d(list_auto_val, data_path_tr,
                                                                                   seg_path_tr)
        inter_cnn_img_train = auto_cnn_img_train + auto_cnn_img_val
        inter_cnn_labels_train = auto_cnn_labels_train + auto_cnn_labels_val
        inter_cnn_img_val, inter_cnn_labels_val = load_prostate_images_and_labels_3d(list_inter_val, data_path_tr,
                                                                                     seg_path_tr)
        test_img, test_labels = load_prostate_images_and_labels_3d_test(list_test, data_path_tr, seg_path_tr)

        # Create the datasets with images and masks
        rot_angle = 5
        max_crop = 1.1

        training_dataset_auto_cnn = GetProstateDataset(
            x_data=auto_cnn_img_train, y_data=auto_cnn_labels_train, dimensions=3,
            transform=bool_data_augmentation, rot_angle=rot_angle, max_crop=max_crop, bool_binary=bool_binary_testing)
        validation_dataset_auto_cnn = GetProstateDataset(
            x_data=auto_cnn_img_val, y_data=auto_cnn_labels_val, dimensions=3, transform=False,
            bool_binary=bool_binary_testing)
        training_dataset_inter_cnn = GetProstateDataset(
            x_data=inter_cnn_img_train, y_data=inter_cnn_labels_train, dimensions=3,
            transform=bool_data_augmentation, rot_angle=rot_angle, max_crop=max_crop, bool_binary=bool_binary_testing)
        validation_dataset_inter_cnn = GetProstateDataset(
            x_data=inter_cnn_img_val, y_data=inter_cnn_labels_val, dimensions=3, transform=False,
            bool_binary=bool_binary_testing)
        test_dataset = GetProstateDataset(
            x_data=test_img, y_data=test_labels, dimensions=3, transform=False, bool_binary=bool_binary_testing)

        del data_folder, data_path_tr, seg_path_tr, list_total, list_auto_train, list_auto_val, list_inter_val, \
            list_test, auto_cnn_img_train, auto_cnn_labels_train, auto_cnn_img_val, auto_cnn_labels_val, \
            inter_cnn_img_train, inter_cnn_labels_train, inter_cnn_img_val, inter_cnn_labels_val, test_img, test_labels
    else:
        # Set the location of the data
        volume_size = (160, 192, 160)
        volume_size_string = '_'.join(str(i) for i in volume_size)
        data_file = 'shlynur_brats_data_3D_size_{}_gustavs_split_{}.hdf5'.format(volume_size_string, gustav_split)

        if operating_system == 'linux':
            data_file = os.path.join('/scratch_net/giggo/shlynur/msc/', data_file)
        else:
            data_file = os.path.join(r"D:\ETH_Projects", data_file)

        # Check if the HDF5 file exists (no action needed) or not (error message)
        if not os.path.isfile(data_file):
            raise ValueError('The HDF5 file does not exist.')

        # # Create the datasets with images and masks
        training_dataset_auto_cnn = GetBratsDataset(file_path=data_file, group='autocnn', mode='train',
                                                    transform=bool_data_augmentation)
        validation_dataset_auto_cnn = GetBratsDataset(file_path=data_file, group='autocnn', mode='val',
                                                      transform=False)
        training_dataset_inter_cnn = GetBratsDataset(file_path=data_file, group='intercnn', mode='train',
                                                     transform=bool_data_augmentation)
        validation_dataset_inter_cnn = GetBratsDataset(file_path=data_file, group='intercnn', mode='val',
                                                       transform=False)
        test_dataset = GetBratsDataset(file_path=data_file, mode='test', transform=False)

    # Create the data loaders from the datasets
    dataloader_auto_cnn_train = DataLoader(
        dataset=training_dataset_auto_cnn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_auto_cnn_val = DataLoader(
        dataset=validation_dataset_auto_cnn, batch_size=1, shuffle=True, num_workers=num_workers)
    dataloader_inter_cnn_train = DataLoader(
        dataset=training_dataset_inter_cnn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_inter_cnn_val = DataLoader(
        dataset=validation_dataset_inter_cnn, batch_size=1, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    print("autoCNN_train: \t{} \t\tautoCNN_val: \t{} \ninterCNN_train: {} \t\tinterCNN_val: \t{} \ntesting: \t{}"
          .format(len(training_dataset_auto_cnn), len(validation_dataset_auto_cnn), len(training_dataset_inter_cnn),
                  len(validation_dataset_inter_cnn), len(test_dataset)))

    # Create the networks.
    if model_type == 'gustav':
        cnn_auto = UNet3D(bool_prostate_data=bool_prostate_data, in_channels=channels_auto,
                          number_of_classes=num_classes, number_of_filters=num_filters_auto_cnn,
                          kernel_size=(3, 3, 3), padding=padding, max_pooling_d=max_pooling_d).cuda()
        cnn_inter = UNet3D(bool_prostate_data=bool_prostate_data, in_channels=channels_inter,
                           number_of_classes=num_classes, number_of_filters=num_filters_inter_cnn,
                           kernel_size=(3, 3, 3), padding=padding, max_pooling_d=max_pooling_d).cuda()
    elif model_type == 'robin':
        cnn_auto = Reversible3D(encoder_depth=1).cuda()
        cnn_inter = Reversible3D(encoder_depth=1).cuda()
    else:
        raise ValueError('The variable model_type must be either "gustav" or "robin"')
    # sample_input = (1, 15, 320, 320) if bool_prostate_data else (4, 160, 192, 160)
    # summary(model=cnn_auto, input_size=sample_input)

    # Create the optimizers.
    optimizer_auto = Adam(params=cnn_auto.parameters(), lr=learning_rate)
    scheduler_auto = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_auto, mode='min', factor=0.99,
                                                    patience=100, verbose=True)
    # scheduler_auto = lr_scheduler.CyclicLR(optimizer=optimizer_auto, base_lr=5e-4, max_lr=3e-3)
    optimizer_inter = Adam(params=cnn_inter.parameters(), lr=learning_rate)
    scheduler_inter = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_inter, mode='min', factor=0.99,
                                                     patience=10, verbose=True)
    # scheduler_inter = lr_scheduler.StepLR(optimizer=optimizer_inter, step_size=10, gamma=0.1)
    print_memory_usage(status='Before training everything')

    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    train_activation = nn.Softmax(dim=1)

    # Load a previous autoCNN configuration if the boolean is True.
    if bool_load_previous_checkpoint[0]:
        if os.path.isfile(os.path.join('models', model_name_auto)):
            print('AutoCNN model is in the current folder.')
            load_model_folder[0] = 'models'
            load_model_file[0] = model_name_auto
        cnn_auto, optimizer_auto, scheduler_auto, starting_epoch_auto_cnn, auto_class_scores = load_checkpoint(
            model=cnn_auto, optimizer=optimizer_auto, scheduler=scheduler_auto, file_folder=load_model_folder[0],
            filename=load_model_file[0])
    else:
        starting_epoch_auto_cnn = 0
        auto_class_scores = np.ones(num_classes) * 0.0

    # Load a previous interCNN configuration if the boolean is True.
    if bool_load_previous_checkpoint[1]:
        if os.path.isfile(os.path.join('models', model_name_inter)):
            print('InterCNN model is in the current folder.')
            load_model_folder[1] = 'models'
            load_model_file[1] = model_name_inter
        cnn_inter, optimizer_inter, scheduler_inter, starting_epoch_inter_cnn, inter_class_scores = load_checkpoint(
            model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, file_folder=load_model_folder[1],
            filename=load_model_file[1])
    else:
        starting_epoch_inter_cnn = 0
        inter_class_scores = np.ones(num_classes) * 0.0

    ####################################################################################################################
    # # # # # # # # # # # # # # # # # # # # # # # # # AutoCNN Training.  # # # # # # # # # # # # # # # # # # # # # # # #
    ####################################################################################################################
    # if bool_auto_cnn_train:
    #     print("\n" + "#" * 10 + " auto_CNN training. " + "#" * 10)
    #     # Initialize variables.
    #     loss_list = np.zeros((num_epochs_auto_cnn, 2))
    #     class_score_train_list = np.zeros((num_epochs_auto_cnn, num_classes))
    #     class_score_validation_list = np.zeros((num_epochs_auto_cnn, num_classes))
    #
    #     # Set up the Pandas dataframe.
    #     auto_df = pd.DataFrame(
    #         columns=('Epoch', 'Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy'))
    #     auto_df['Epoch'] = auto_df['Epoch'].astype('int64')
    #     auto_df = auto_df.set_index('Epoch', inplace=False)
    #
    #     auto_cnn_timer = time.time()
    #     for epoch in range(starting_epoch_auto_cnn, num_epochs_auto_cnn):
    #         running_loss = 0
    #         train_average_classes, train_mean_average_classes = [], []
    #         for images, labels in dataloader_auto_cnn_train:
    #             cnn_auto.train()
    #
    #             if bool_prostate_data:
    #                 images = images.unsqueeze(1)
    #             # else:
    #                 # # only use one channel of the BraTS dataset.
    #                 # if channels_auto == 1:
    #                 #     images = images[:, 3, :, :, :]
    #                 #     images = images.unsqueeze(1)
    #
    #             images = images.float().cuda()
    #             labels = labels.long().cuda()
    #
    #             # Forward + Backward + Optimizer
    #             optimizer_auto.zero_grad()
    #             outputs = cnn_auto(images)
    #             del images
    #             loss = criterion(outputs, labels)
    #             # loss = cross_entropy3d(network_output=outputs, target_output=labels, weight=loss_weight,
    #             #                        size_average=False)
    #             # loss = brats_dice_loss(outputs=outputs, labels=labels, non_squared=False)
    #             # loss = brats_dice_loss_original_4(outputs=outputs, labels=labels, non_squared=False)
    #             loss.backward()
    #             # Weight decay stuff
    #             for group in optimizer_auto.param_groups:
    #                 for param in group['params']:
    #                     param.data = param.data.add(-lr_wd * group['lr'], param.data)
    #             optimizer_auto.step()
    #             running_loss += loss.item()
    #
    #             # Training accuracy calculations
    #             cnn_auto.eval()
    #             train_batch_score = []
    #             train_max_in = train_activation(outputs)
    #             train_max_in = torch.max(train_max_in, 1)[1]
    #             # train_max_in = train_max_in.type(torch.LongTensor)
    #             for b in range(labels.shape[0]):
    #                 train_classes_score = class_dice_3d_prostate(target=labels[b, :, :, :], prediction=train_max_in[b, :, :, :],
    #                                                              class_num=num_classes)
    #                 train_batch_score.append(train_classes_score)
    #             train_average_classes.append(train_batch_score)
    #
    #             del labels, outputs, train_activation, train_max_in
    #
    #         train_average_classes = np.squeeze(np.array(train_average_classes))
    #         train_mean_average_classes = np.mean(train_average_classes, axis=0)
    #         if len(dataloader_auto_cnn_train.dataset) == 1:
    #             class_scores_training = train_average_classes
    #         else:
    #             class_scores_training = train_mean_average_classes
    #         class_score_train_list[epoch, :] = class_scores_training
    #
    #         # Validation accuracy / convergence
    #         class_scores_validating, val_loss = convergence_check_auto_cnn(
    #             cnn=cnn_auto, bool_prostate_data=bool_prostate_data, number_of_classes=num_classes,
    #             data_loader=dataloader_auto_cnn_val)
    #         class_score_validation_list[epoch, :] = class_scores_validating
    #         # convergence_list[epoch] = np.mean(class_scores_validating[1:])
    #
    #         scheduler_auto.step(val_loss)
    #
    #         train_loss = running_loss / len(dataloader_auto_cnn_train)
    #         loss_list[epoch, 0] = train_loss
    #         loss_list[epoch, 1] = val_loss
    #
    #         new_df = pd.Series(data={'Training loss': train_loss, 'Validation loss': val_loss,
    #                                  'Training accuracy': np.mean(class_scores_training[1:]),
    #                                  'Validation accuracy': np.mean(class_scores_validating[1:])}, name=(epoch + 1))
    #         auto_df = auto_df.append(new_df)
    #         auto_df.to_csv('autoCNN_dataframe.csv')
    #
    #         print_current_epoch_results(current_epoch=epoch, total_epochs=num_epochs_auto_cnn, training_loss=train_loss,
    #                                     validation_loss=val_loss, training_accuracy=class_scores_training,
    #                                     validation_accuracy=class_scores_validating,
    #                                     bool_prostate_data=bool_prostate_data, bool_binary_testing=bool_binary_testing)
    #
    #         # if bool_prostate_data:
    #         #     if inter_scribble_mode == '2d':
    #         #         print("Epoch {:3d} / {} \tTrain / val loss: {:.4f} / {:.4f} \tTrain / val acc: [{:.4f}, {:.4f}, "
    #         #               "{:.4f}] / [{:.4f}, {:.4f}, {:.4f}] \tSignificant convergence: {:.4f}"
    #         #               .format(epoch + 1, num_epochs_auto_cnn, train_loss, val_loss, class_scores_training[0],
    #         #                       class_scores_training[1], class_scores_training[2], class_scores_validating[0],
    #         #                       class_scores_validating[1], class_scores_validating[2],
    #         #                       np.mean(class_scores_validating[1:])))
    #         #     else:
    #         #         print("Epoch {:3d} / {} \tTrain / val loss: {:.4f} / {:.4f} \tTrain / val acc: [{:.4f}, {:.4f}] "
    #         #               " / [{:.4f}, {:.4f}] \tSignificant convergence: {:.4f}"
    #         #               .format(epoch + 1, num_epochs_auto_cnn, train_loss, val_loss, class_scores_training[0],
    #         #                       class_scores_training[1], class_scores_validating[0], class_scores_validating[1],
    #         #                       np.mean(class_scores_validating[1:])))
    #         # else:
    #         #     print("Epoch {:3d} / {} \tTrain / val loss: {:.4f} / {:.4f} \tTrain / val acc: [{:.4f}, {:.4f}, "
    #         #           "{:.4f}, {:.4f}] / [{:.4f}, {:.4f}, {:.4f}, {:.4f}] \tSignificant convergence: {:.4f}"
    #         #           .format(epoch + 1, num_epochs_auto_cnn, train_loss, val_loss, class_scores_training[0],
    #         #                   class_scores_training[1], class_scores_training[2], class_scores_training[3],
    #         #                   class_scores_validating[0], class_scores_validating[1], class_scores_validating[2],
    #         #                   class_scores_validating[3], np.mean(class_scores_validating[1:])))
    #         # # list(np.around(class_scores_training, 4)), list(np.around(class_scores_validating, 4)),
    #
    #         # Save the parameters
    #         if bool_save_results:
    #             # Save the CNN with the best validation score
    #             if np.mean(class_scores_validating[1:]) > np.mean(auto_class_scores[1:]):
    #                 print("Model checkpoint at epoch {}. Old accuracy: {:.4f}, new accuracy: {:.4f}."
    #                       .format((epoch + 1), np.mean(auto_class_scores[1:]), np.mean(class_scores_validating[1:])))
    #                 auto_class_scores = class_scores_validating
    #                 save_checkpoint(model=cnn_auto, optimizer=optimizer_auto, scheduler=scheduler_auto, epoch=epoch,
    #                                 val_score=auto_class_scores, model_name=os.path.join('models', model_name_auto))
    #     print_time(auto_cnn_timer)
    #
    #     auto_loss_image_name = \
    #         'autocnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_train_val_losses' \
    #         .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
    #                 num_filters_auto_cnn, learning_rate, bool_data_augmentation)
    #     plot_loss_images(loss_list, filename=auto_loss_image_name, mode='auto')
    #
    #     if bool_save_results:
    #         save_str = 'autocnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_' \
    #             .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
    #                     num_filters_auto_cnn, learning_rate, bool_data_augmentation)
    #         np.save(save_str + 'train_scores.npy', class_score_train_list)
    #         np.save(save_str + 'val_scores.npy', class_score_validation_list)
    #         np.save(save_str + 'loss_list.npy', loss_list)
    #
    #         save_image_name = save_str[:-1]
    #         plot_accuracy_images(model='inter', filename=save_image_name, class_num=num_classes,
    #                              training_list=class_score_train_list, validation_list=class_score_validation_list)
    #
    #         # Update the epoch parameter in the saved torch dictionary.
    #         cnn_auto, optimizer_auto, scheduler_auto, starting_epoch_auto_cnn, auto_class_scores = load_checkpoint(
    #             model=cnn_auto, optimizer=optimizer_auto, scheduler=scheduler_auto, file_folder='models',
    #             filename=model_name_auto)
    #         save_checkpoint(model=cnn_auto, optimizer=optimizer_auto, scheduler=scheduler_auto,
    #                         epoch=num_epochs_auto_cnn, val_score=auto_class_scores,
    #                         model_name=os.path.join('models', model_name_auto))
    #     # summary(model=cnn_auto, input_size=sample_input)

    # Load the best autoCNN model.
    if os.path.isfile(os.path.join('models', model_name_auto)):
        cnn_auto.load_state_dict(torch.load(os.path.join('models', model_name_auto))['state_dict'])
    else:
        cnn_auto, optimizer_auto, scheduler_auto, starting_epoch_auto_cnn, auto_class_scores = load_checkpoint(
            model=cnn_auto, optimizer=optimizer_auto, scheduler=scheduler_auto, file_folder=load_model_folder[0],
            filename=load_model_file[0])
    cnn_auto.eval()
    print_memory_usage(status='After loading the auto_cnn model')

    ####################################################################################################################
    # # # # # # # # # # # # # # # # # # # # # # # # # InterCNN Training. # # # # # # # # # # # # # # # # # # # # # # # #
    ####################################################################################################################
    # if bool_inter_cnn_train:
    #     print("\n" + "#" * 10 + " inter_CNN training. " + "#" * 10)
    #     # Initialize variables.
    #     loss_list = np.zeros((num_epochs_inter_cnn, 2))
    #     class_scores_auto_cnn_val_list = []
    #     # class_scores_inter_cnn_train_list, class_scores_inter_cnn_val_list = [], []
    #     class_scores_inter_cnn_train_list = np.zeros((num_epochs_inter_cnn, num_classes))
    #     class_scores_inter_cnn_val_list = np.zeros((num_epochs_inter_cnn, num_classes))
    #     convergence_list = np.zeros(num_epochs_inter_cnn)
    #     inter_cnn_timer = time.time()
    #     iterations_scores = 0
    #     accumulation_steps = 2 if bool_loss_summation else 1
    #
    #     # Set up the Pandas dataframe.
    #     inter_df = pd.DataFrame(
    #         columns=('Epoch', 'Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy'))
    #     inter_df['Epoch'] = inter_df['Epoch'].astype('int64')
    #     inter_df = inter_df.set_index('Epoch', inplace=False)
    #
    #     for epoch in range(starting_epoch_inter_cnn, num_epochs_inter_cnn):
    #         running_loss = 0
    #         counter = 0
    #         train_average_classes = []
    #         for i, (images, labels) in enumerate(dataloader_inter_cnn_train):
    #             cnn_auto.eval()
    #             cnn_inter.train()
    #             train_class_scores_inter_cnn_iterations = []
    #             if bool_prostate_data:
    #                 images = images.unsqueeze(1)
    #
    #             # images_copy = images.float()
    #             labels_copy = labels.long()
    #             images = images.float().cuda()
    #             labels = labels.long().cuda()
    #
    #             # autoCNN prediction
    #             outputs_auto = cnn_auto(images)
    #             prediction_auto = prediction_converter(outputs_auto)
    #             if bool_loss_summation:
    #                 prediction_auto_no_max = train_activation(outputs_auto)  # .detach().clone())
    #                 prediction_torch = prediction_auto_no_max  # .unsqueeze(1)
    #             else:
    #                 prediction_torch = prediction_auto.unsqueeze(1)
    #             del outputs_auto
    #
    #             incorrect_predictions, new_labels = identify_incorrect_pixels(
    #                 target=labels_copy, prediction=prediction_auto, class_num=num_classes)
    #             scribbles = mask_creator(incorrect=incorrect_predictions, label=new_labels, class_num=num_classes,
    #                                      number_of_slices=inter_number_of_slices, mode=inter_scribble_mode,
    #                                      number_of_scribbles=number_of_scribbles, size_of_scribble=size_of_scribble,
    #                                      bool_best_placement=bool_best_placement)
    #             # print("Prediction: {}, Incorrect predictions: {}, Incorrect labels: {}, Scribbles: {}".format(
    #             #     prediction_auto.shape, incorrect_predictions.shape, new_labels.shape, scribbles.shape))
    #
    #             prediction_torch = prediction_torch.float().cuda()
    #             scribbles_torch = torch.from_numpy(scribbles)
    #             scribbles_torch = scribbles_torch.float().cuda()
    #
    #             loss_adder = torch.zeros(1)
    #
    #             optimizer_inter.zero_grad()
    #             for j in range(max_iterations_train):
    #                 # print("InterCNN iteration {}. \tImages: {} \tprediction: {} \tscribbles: {}"
    #                 #       .format(j, images.shape, prediction_torch.shape, scribbles_torch.shape))
    #                 # print_memory_usage('Iteration {}'.format(j))
    #                 network_inputs = torch.cat((images, prediction_torch, scribbles_torch), dim=1)
    #                 del prediction_torch, scribbles_torch
    #                 outputs_inter = cnn_inter(network_inputs)
    #                 del network_inputs
    #
    #                 prediction_inter = prediction_converter(outputs_inter)
    #                 if bool_loss_summation:
    #                     prediction_inter_no_max = train_activation(outputs_inter)  # .detach().clone())
    #                     prediction_torch = prediction_inter_no_max  # .unsqueeze(1)
    #                 else:
    #                     prediction_torch = prediction_inter.unsqueeze(1)
    #
    #                 loss = criterion(outputs_inter, labels)
    #
    #                 del outputs_inter
    #
    #                 # loss = cross_entropy3d(network_output=outputs_inter, target_output=labels, weight=loss_weight,
    #                 #                        size_average=False)
    #                 # if accumulation_steps > 1:
    #                 #     loss = loss / accumulation_steps
    #                 # loss.backward()
    #                 # running_loss += loss.item()
    #                 # counter += 1
    #
    #                 if bool_loss_summation:
    #                     loss_adder += loss
    #
    #                     # Backpropagate, calculate the loss and update the model parameters.
    #                     if (j + 1) % accumulation_steps == 0:
    #                         loss_adder.backward()
    #                         running_loss += loss_adder.item()
    #                         counter += 1
    #
    #                         # Weight decay stuff
    #                         for group in optimizer_inter.param_groups:
    #                             for param in group['params']:
    #                                 param.data = param.data.add(-lr_wd * group['lr'], param.data)
    #                         optimizer_inter.step()
    #                         optimizer_inter.zero_grad()
    #                         loss_adder = 0
    #                         # scribbles = 0
    #                 else:
    #                     loss.backward()
    #                     running_loss += loss.item()
    #                     counter += 1
    #
    #                     # Weight decay stuff
    #                     for group in optimizer_inter.param_groups:
    #                         for param in group['params']:
    #                             param.data = param.data.add(-lr_wd * group['lr'], param.data)
    #                     optimizer_inter.step()
    #                     # running_loss += loss.item()
    #                     optimizer_inter.zero_grad()
    #
    #                 incorrect_predictions, new_labels = identify_incorrect_pixels(
    #                     target=labels_copy, prediction=prediction_inter, class_num=num_classes)
    #                 scribbles = scribbles + mask_creator(
    #                     incorrect=incorrect_predictions, label=new_labels, class_num=num_classes,
    #                     number_of_slices=inter_number_of_slices, mode=inter_scribble_mode,
    #                     number_of_scribbles=number_of_scribbles, size_of_scribble=size_of_scribble,
    #                     bool_best_placement=bool_best_placement)
    #
    #                 prediction_torch = prediction_torch.float().cuda()
    #                 scribbles_torch = torch.from_numpy(scribbles)
    #                 scribbles_torch = scribbles_torch.float().cuda()
    #
    #                 # Training accuracy
    #                 cnn_inter.eval()
    #                 train_batch_score = []
    #                 # train_max_in = train_activation(outputs_inter)
    #                 # train_max_in = torch.max(train_max_in, 1)[1]
    #
    #                 for b in range(labels.shape[0]):
    #                     train_classes_score = class_dice_3d_prostate(target=labels[b, :, :, :],
    #                                                                  prediction=prediction_inter[b, :, :, :],
    #                                                                  class_num=num_classes)
    #                     train_batch_score.append(train_classes_score)
    #                 train_class_scores_inter_cnn_iterations.append(train_batch_score)
    #                 cnn_inter.train()
    #
    #                 # del outputs_inter
    #
    #             train_average_classes.append(train_class_scores_inter_cnn_iterations)
    #
    #             del images, labels, prediction_torch, scribbles_torch
    #
    #         train_average_classes = np.squeeze(np.array(train_average_classes))
    #         train_average_classes = np.mean(train_average_classes, axis=0)
    #         if max_iterations_train == 1 or len(dataloader_inter_cnn_train.dataset) == 1:
    #             train_inter_cnn_score = train_average_classes
    #         else:
    #             train_inter_cnn_score = np.mean(train_average_classes, axis=0)
    #
    #         # figure, axes = plt.subplots(nrows=2, ncols=3, sharex='all', figsize=(19.2, 10.8))
    #         # random_slice = np.random.randint(labels.shape[1])
    #         # axes = np.ravel(axes)
    #         # axes[0].imshow(images.cpu()[0, 0, random_slice, :, :])
    #         # axes[0].axis('off')
    #         # axes[1].imshow(labels.cpu()[0, random_slice, :, :])
    #         # axes[1].axis('off')
    #         # axes[2].imshow(prediction_torch.cpu()[0, 0, random_slice, :, :])
    #         # axes[2].axis('off')
    #         # axes[3].imshow(scribbles_torch.cpu()[0, 0, random_slice, :, :])
    #         # axes[3].axis('off')
    #         # axes[4].imshow(scribbles_torch.cpu()[0, 1, random_slice, :, :])
    #         # axes[4].axis('off')
    #         # axes[5].imshow(scribbles_torch.cpu()[0, 2, random_slice, :, :])
    #         # axes[5].axis('off')
    #         # # figure.savefig("testing_images/inter_cnn_3D__epoch_{}.png".format(epoch), bbox_inches='tight')
    #         # figure.savefig('testing_images' + model_name_inter[:-3] + '.png', bbox_inches='tight')
    #         # plt.close(figure)
    #
    #         # figure_inter_cnn = "intercnn_figure_epoch_{}".format(epoch + 1)
    #         # figure_inter_cnn_filename = os.path.join('intercnn_images', figure_inter_cnn)
    #         # plot_inter_cnn_images(images=images_copy, labels=labels_copy, prediction=prediction_inter,
    #         #                       incorrect_predictions=incorrect_predictions, new_labels=new_labels,
    #         #                       scribbles=scribbles, filename=figure_inter_cnn_filename)
    #
    #         class_scores_auto_new, class_scores_inter_new, val_loss, new_iterations_scores = \
    #             accuracy_check_auto_inter_cnn(
    #                 auto_cnn=cnn_auto, inter_cnn=cnn_inter, class_num=num_classes, max_iterations=max_iterations_train,
    #                 bool_prostate_data=bool_prostate_data, data_loader=dataloader_inter_cnn_val,
    #                 number_of_slices=inter_number_of_slices, save_folder=save_folder, model_type=model_type,
    #                 num_epochs_inter_cnn=num_epochs_inter_cnn, num_filters_inter_cnn=num_filters_inter_cnn,
    #                 learning_rate=learning_rate, bool_data_augmentation=bool_data_augmentation,
    #                 bool_loss_summation=bool_loss_summation, accumulation_steps=accumulation_steps,
    #                 mode=inter_scribble_mode, number_of_scribbles=number_of_scribbles,
    #                 size_of_scribble=size_of_scribble, bool_best_placement=bool_best_placement)
    #
    #         scheduler_inter.step(val_loss)
    #
    #         # class_scores_inter_cnn_train_list.append(train_inter_cnn_score)
    #         class_scores_inter_cnn_train_list[epoch, :] = train_inter_cnn_score
    #         class_scores_auto_cnn_val_list.append(class_scores_auto_new)
    #         # class_scores_inter_cnn_val_list.append(class_scores_inter_new)
    #         class_scores_inter_cnn_val_list[epoch, :] = class_scores_inter_new
    #         convergence_list[epoch] = np.mean(class_scores_inter_new)
    #
    #         train_loss = running_loss / counter
    #         loss_list[epoch, 0] = train_loss
    #         loss_list[epoch, 1] = val_loss
    #
    #         new_df = pd.Series(data={'Training loss': train_loss, 'Validation loss': val_loss,
    #                                  'Training accuracy': np.mean(train_inter_cnn_score[1:]),
    #                                  'Validation accuracy': np.mean(class_scores_inter_new[1:])}, name=(epoch + 1))
    #         inter_df = inter_df.append(new_df)
    #         inter_df.to_csv('intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_dataframe.csv'
    #                         .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
    #                                 num_filters_inter_cnn, learning_rate, bool_data_augmentation))
    #
    #         print_current_epoch_results(current_epoch=epoch, total_epochs=num_epochs_inter_cnn,
    #                                     training_loss=train_loss, validation_loss=val_loss,
    #                                     training_accuracy=train_inter_cnn_score,
    #                                     validation_accuracy=class_scores_inter_new,
    #                                     bool_prostate_data=bool_prostate_data, bool_binary_testing=bool_binary_testing)
    #
    #         # if bool_prostate_data:
    #         #     print("Epoch {:3d} / {} \tTrain / val loss: {:.4f} / {:.4f} \tTrain / val acc: [{:.4f}, {:.4f}]"
    #         #           " / [{:.4f}, {:.4f}] \tSignificant convergence: {:.4f}"
    #         #           .format(epoch + 1, num_epochs_inter_cnn, train_loss, val_loss, train_inter_cnn_score[0],
    #         #                   train_inter_cnn_score[1],  # train_inter_cnn_score[2],
    #         #                   class_scores_inter_new[0],
    #         #                   class_scores_inter_new[1],  # class_scores_inter_new[2],
    #         #                   np.mean(class_scores_inter_new[1:])))
    #         # else:
    #         #     print("Epoch {:3d} / {} \tTrain / val loss: {:.4f} / {:.4f} \tTrain / val acc: [{:.4f}, {:.4f}, "
    #         #           "{:.4f}, {:.4f}] / [{:.4f}, {:.4f}, {:.4f}, {:.4f}] \tSignificant convergence: {:.4f}"
    #         #           .format(epoch + 1, num_epochs_inter_cnn, train_loss, val_loss, train_inter_cnn_score[0],
    #         #                   train_inter_cnn_score[1], train_inter_cnn_score[2], train_inter_cnn_score[3],
    #         #                   class_scores_inter_new[0], class_scores_inter_new[1], class_scores_inter_new[2],
    #         #                   class_scores_inter_new[3], np.mean(class_scores_inter_new[1:])))
    #
    #         # print("Epoch {:3d} / {} \tLoss: {:0.8f} \tTraining acc: {} \tConvergence: {} \t"
    #         #       "Significant convergence: {:.4f}"
    #         #       .format(epoch + 1, num_epochs_inter_cnn, train_loss, np.around(train_inter_cnn_score, 4),
    #         #               np.around(class_scores_inter_new, 4), np.mean(class_scores_inter_new[1:])))
    #
    #         if bool_save_results:
    #             # Save the CNN with the highest Dice score
    #             if np.mean(class_scores_inter_new[1:]) > np.mean(inter_class_scores[1:]):
    #                 print("Model checkpoint at epoch {}. Old accuracy: {:.4f}, new accuracy: {:.4f}."
    #                       .format((epoch + 1), np.mean(inter_class_scores[1:]), np.mean(class_scores_inter_new[1:])))
    #                 inter_class_scores = class_scores_inter_new
    #                 iterations_scores = new_iterations_scores
    #                 save_checkpoint(model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, epoch=epoch,
    #                                 val_score=inter_class_scores, model_name=os.path.join('models', model_name_inter))
    #     print_time(inter_cnn_timer)
    #
    #     inter_loss_image_name = \
    #         'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_train_val_losses'\
    #         .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
    #                 num_filters_inter_cnn, learning_rate, bool_data_augmentation)
    #     plot_loss_images(loss_list, filename=inter_loss_image_name, mode='inter')
    #
    #     if bool_save_results:
    #         save_str = 'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_' \
    #             .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
    #                     num_filters_inter_cnn, learning_rate, bool_data_augmentation)
    #         np.save(save_str + 'auto_val_scores.npy', class_scores_auto_cnn_val_list)
    #         np.save(save_str + 'inter_val_scores.npy', class_scores_inter_cnn_val_list)
    #         np.save(save_str + 'loss_list.npy', loss_list)
    #         np.save(save_str + 'val_iterations.npy', iterations_scores)
    #
    #         save_image_name = save_str[:-1]
    #         plot_accuracy_images(
    #             model='inter', filename=save_image_name, class_num=num_classes,
    #             training_list=class_scores_inter_cnn_train_list, validation_list=class_scores_inter_cnn_val_list)
    #
    #         # Update the epoch parameter in the saved torch dictionary.
    #         cnn_inter, optimizer_inter, scheduler_inter, starting_epoch_inter_cnn, inter_class_scores = load_checkpoint(
    #             model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, file_folder='models',
    #             filename=model_name_inter)
    #         save_checkpoint(model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter,
    #                         epoch=num_epochs_inter_cnn, val_score=inter_class_scores,
    #                         model_name=os.path.join('models', model_name_inter))

    # Load the best interCNN model.
    if os.path.isfile(os.path.join('models', model_name_inter)):
        cnn_inter.load_state_dict(torch.load(os.path.join('models', model_name_inter))['state_dict'])
    else:
        cnn_inter, optimizer_inter, scheduler_inter, starting_epoch_inter_cnn, inter_class_scores = load_checkpoint(
            model=cnn_inter, optimizer=optimizer_inter, scheduler=scheduler_inter, file_folder=load_model_folder[1],
            filename=load_model_file[1])
    cnn_inter.eval()
    print_memory_usage(status='After loading the inter_cnn model')

    ####################################################################################################################
    # # # # # # # # # # # # # # # # # # # # # # # Testing both networks. # # # # # # # # # # # # # # # # # # # # # # # #
    ####################################################################################################################
    print("\n" + "#" * 10 + " auto_CNN and inter_CNN testing. " + "#" * 10)
    cnn_auto.eval()
    cnn_inter.eval()
    class_scores_auto_cnn_2d, class_scores_auto_cnn_3d = [], []
    class_scores_inter_cnn_2d, class_scores_inter_cnn_3d = [], []
    hd_scores_auto_cnn, hd95_scores_auto_cnn = [], []
    hd_scores_inter_cnn, hd95_scores_inter_cnn = [], []
    for i, (images, labels) in enumerate(dataloader_test):
        class_scores_inter_cnn_iterations_2d, class_scores_inter_cnn_iterations_3d = [], []
        hd_scores_inter_cnn_iterations, hd95_scores_inter_cnn_iterations = [], []
        if bool_prostate_data:
            images = images.unsqueeze(1)

        images = images.float().cuda()
        labels = labels.long()

        if bool_prostate_data:
            margin = images.shape[2] - 15
            if margin > 0:
                total_output_auto = torch.zeros((1, num_classes, labels.shape[1], labels.shape[2], labels.shape[3]))
                output_auto_tracker = torch.zeros_like(total_output_auto)
                for x in range(margin + 1):
                    # print_memory_usage('Loop with x = {}'.format(x))
                    new_image = images[:, :, x: x + 15, :]  # .cuda()
                    new_output = cnn_auto(new_image).cpu().data  # .numpy()
                    total_output_auto[:, :, x: x + 15, :] += new_output
                    output_auto_tracker[:, :, x: x+15, :] += torch.ones_like(new_output)
                    # print_memory_usage('x = {}'.format(x))
                    del new_image, new_output
                    # print_memory_usage('After deleting')
                outputs_auto = total_output_auto / output_auto_tracker
            else:
                outputs_auto = cnn_auto(images)
        else:
            outputs_auto = cnn_auto(images)
        prediction_auto = prediction_converter(outputs=outputs_auto)

        if bool_loss_summation:
            prediction_auto_no_max = train_activation(outputs_auto)  # .detach().clone())
            prediction_torch = prediction_auto_no_max  # .unsqueeze(1)
        else:
            prediction_torch = prediction_auto.unsqueeze(1)
        del outputs_auto

        # Calculate class scores for autoCNN
        class_scores_auto_new_2d = class_dice_3d_prostate(target=labels[0, :, :, :], prediction=prediction_auto[0, :, :, :],
                                                          class_num=num_classes, mode='2d')
        class_scores_auto_cnn_2d.append(class_scores_auto_new_2d)
        class_scores_auto_new_3d = class_dice_3d_prostate(target=labels[0, :, :, :], prediction=prediction_auto[0, :, :, :],
                                                          class_num=num_classes, mode='3d')
        class_scores_auto_cnn_3d.append(class_scores_auto_new_3d)

        # Calculate HD and HD95 for autoCNN
        hd_scores_auto_new, hd_95_auto_new = class_hd_hd95(
            target=labels[0, :, :, :], prediction=prediction_auto[0, :, :, :], class_num=num_classes)
        hd_scores_auto_cnn.append(hd_scores_auto_new)
        hd95_scores_auto_cnn.append(hd_95_auto_new)

        # Create scribbles for input to interCNN
        incorrect_predictions, new_labels = identify_incorrect_pixels(
            target=labels, prediction=prediction_auto, class_num=num_classes)
        scribbles = mask_creator(
            incorrect=incorrect_predictions, label=new_labels, class_num=num_classes,
            number_of_slices=inter_number_of_slices, mode=inter_scribble_mode, number_of_scribbles=number_of_scribbles,
            size_of_scribble=size_of_scribble, bool_best_placement=bool_best_placement)

        prediction_torch = prediction_torch.float().cuda()
        scribbles_torch = torch.from_numpy(scribbles)
        scribbles_torch = scribbles_torch.float().cuda()

        for index in range(max_iterations_test):

            if bool_prostate_data:
                margin = images.shape[2] - 15
                if margin > 0:
                    total_output_inter = torch.zeros((1, num_classes, labels.shape[1], labels.shape[2],
                                                      labels.shape[3]))
                    output_inter_tracker = torch.zeros_like(total_output_inter)
                    for x in range(margin + 1):
                        # print_memory_usage('Loop with x = {}'.format(x))
                        new_image = images[:, :, x: x + 15, :]
                        new_prediction = prediction_torch[:, :, x: x + 15, :]
                        new_scribble = scribbles_torch[:, :, x: x + 15, :]
                        # print(new_image.shape, new_prediction.shape, new_scribble.shape)
                        network_inputs = torch.cat((new_image, new_prediction, new_scribble), dim=1)
                        new_output = cnn_inter(network_inputs).cpu().data
                        total_output_inter[:, :, x: x + 15, :] += new_output
                        output_inter_tracker[:, :, x: x + 15, :] += torch.ones_like(new_output)
                        # print_memory_usage('Before deleting')
                        del new_image, new_prediction, new_scribble, network_inputs, new_output
                        # print_memory_usage(' After deleting')
                    outputs_inter = total_output_inter / output_inter_tracker
                else:
                    network_inputs = torch.cat((images, prediction_torch, scribbles_torch), dim=1)
                    outputs_inter = cnn_inter(network_inputs)
            else:
                network_inputs = torch.cat((images, prediction_torch, scribbles_torch), dim=1)
                outputs_inter = cnn_inter(network_inputs)
            del prediction_torch, scribbles_torch
            prediction_inter = prediction_converter(outputs=outputs_inter)
            if bool_loss_summation:
                prediction_inter_no_max = train_activation(outputs_inter)  # .detach().clone())
                prediction_torch = prediction_inter_no_max  # .unsqueeze(1)
            else:
                prediction_torch = prediction_inter.unsqueeze(1)
            del outputs_inter

            # Calculate class scores for interCNN
            class_scores_inter_new = class_dice_3d_prostate(
                target=labels[0, :, :, :], prediction=prediction_inter[0, :, :, :], class_num=num_classes, mode='2d')
            class_scores_inter_cnn_iterations_2d.append(class_scores_inter_new)
            class_scores_inter_new = class_dice_3d_prostate(
                target=labels[0, :, :, :], prediction=prediction_inter[0, :, :, :], class_num=num_classes, mode='3d')
            class_scores_inter_cnn_iterations_3d.append(class_scores_inter_new)

            # Calculate HD and HD95 for interCNN
            hd_scores_inter_new, hd_95_inter_new = class_hd_hd95(
                target=labels[0, :, :, :], prediction=prediction_inter[0, :, :, :], class_num=num_classes)
            hd_scores_inter_cnn_iterations.append(hd_scores_inter_new)
            hd95_scores_inter_cnn_iterations.append(hd_95_inter_new)

            # Create new scribbles.
            incorrect_predictions, new_labels = identify_incorrect_pixels(
                target=labels, prediction=prediction_inter, class_num=num_classes)
            scribbles = scribbles + mask_creator(
                incorrect=incorrect_predictions, label=new_labels, class_num=num_classes,
                number_of_slices=inter_number_of_slices, mode=inter_scribble_mode,
                number_of_scribbles=number_of_scribbles, size_of_scribble=size_of_scribble,
                bool_best_placement=bool_best_placement)

            prediction_torch = prediction_torch.float().cuda()
            scribbles_torch = torch.from_numpy(scribbles)
            scribbles_torch = scribbles_torch.float().cuda()

        class_scores_inter_cnn_2d.append(class_scores_inter_cnn_iterations_2d)
        class_scores_inter_cnn_3d.append(class_scores_inter_cnn_iterations_3d)
        hd_scores_inter_cnn.append(hd_scores_inter_cnn_iterations)
        hd95_scores_inter_cnn.append(hd95_scores_inter_cnn_iterations)
        del images, prediction_torch, scribbles_torch

        iterations_str_2d = 'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_test_scores_2d_' \
                            'sample_{}' \
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                    num_filters_inter_cnn, learning_rate, bool_data_augmentation, i)
        iterations_str_3d = 'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_test_scores_3d_' \
                            'sample_{}' \
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                    num_filters_inter_cnn, learning_rate, bool_data_augmentation, i)
        iterations_str_hd = 'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_test_scores_hd_' \
                            'sample_{}' \
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                    num_filters_inter_cnn, learning_rate, bool_data_augmentation, i)
        iterations_str_hd95 = 'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_test_scores_hd_95' \
                              'sample_{}' \
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                    num_filters_inter_cnn, learning_rate, bool_data_augmentation, i)

        np.save(os.path.join('iterations', iterations_str_2d), class_scores_inter_cnn_iterations_2d)
        np.save(os.path.join('iterations', iterations_str_3d), class_scores_inter_cnn_iterations_3d)
        np.save(os.path.join('iterations', iterations_str_hd), hd_scores_inter_cnn_iterations)
        np.save(os.path.join('iterations', iterations_str_hd95), hd95_scores_inter_cnn_iterations)

        # np.save(
        #     os.path.join('iterations',
        #                  'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_test_scores_sample_{}'
        #                  .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
        #                          num_filters_inter_cnn, learning_rate, bool_data_augmentation, i)),
        #     class_scores_inter_cnn_iterations)

        print("Sample {:2d} / {} \t2D autoCNN/interCNN accuracy: {} / {} "
              "\tSignificant 2D autoCNN/interCNN accuracy: {:.4f} / {:.4f}"
              .format(i + 1, len(dataloader_test.dataset), np.around(class_scores_auto_new_2d, 4),
                      np.around(np.mean(class_scores_inter_cnn_iterations_2d, axis=0), 4),
                      np.mean(class_scores_auto_new_2d[1:]),
                      np.mean(np.mean(class_scores_inter_cnn_iterations_2d, axis=0)[1:])))
        print("\t\t3D autoCNN/interCNN accuracy: {} / {} \tSignificant 3D autoCNN/interCNN accuracy: {:.4f} / {:.4f}"
              .format(np.around(class_scores_auto_new_3d, 4),
                      np.around(np.mean(class_scores_inter_cnn_iterations_3d, axis=0), 4),
                      np.mean(class_scores_auto_new_3d[1:]),
                      np.mean(np.mean(class_scores_inter_cnn_iterations_3d, axis=0)[1:])))

        print("\t\tHD autoCNN/interCNN accuracy: {} / {} \tSignificant 2D autoCNN/interCNN accuracy: {:.4f} / {:.4f}"
              .format(np.around(hd_scores_auto_new, 4), np.around(np.mean(hd_scores_inter_cnn_iterations, axis=0), 4),
                      np.mean(hd_scores_auto_new[1:]), np.mean(np.mean(hd_scores_inter_cnn_iterations, axis=0)[1:])))
        print("\t\tHD95 autoCNN/interCNN accuracy: {} / {} \tSignificant 3D autoCNN/interCNN accuracy: {:.4f} / {:.4f}"
              .format(np.around(hd_95_auto_new, 4), np.around(np.mean(hd95_scores_inter_cnn_iterations, axis=0), 4),
                      np.mean(hd_95_auto_new[1:]), np.mean(np.mean(hd95_scores_inter_cnn_iterations, axis=0)[1:])))

    class_scores_auto_cnn_2d = np.squeeze(np.array(class_scores_auto_cnn_2d))
    class_scores_auto_cnn_3d = np.squeeze(np.array(class_scores_auto_cnn_3d))
    class_scores_inter_cnn_2d = np.squeeze(np.array(class_scores_inter_cnn_2d))
    class_scores_inter_cnn_3d = np.squeeze(np.array(class_scores_inter_cnn_3d))

    hd_scores_auto_cnn = np.squeeze(np.array(hd_scores_auto_cnn))
    hd95_scores_auto_cnn = np.squeeze(np.array(hd95_scores_auto_cnn))
    hd_scores_inter_cnn = np.squeeze(np.array(hd_scores_inter_cnn))
    hd95_scores_inter_cnn = np.squeeze(np.array(hd95_scores_inter_cnn))

    mean_auto_cnn_2d = np.mean(class_scores_auto_cnn_2d, axis=0)
    mean_auto_cnn_3d = np.mean(class_scores_auto_cnn_3d, axis=0)
    mean_hd_auto_cnn = np.mean(hd_scores_auto_cnn, axis=0)
    mean_hd95_auto_cnn = np.mean(hd95_scores_auto_cnn, axis=0)
    if max_iterations_test == 1:
        mean_inter_cnn_2d = np.mean(class_scores_inter_cnn_2d, axis=0)
        mean_inter_cnn_3d = np.mean(class_scores_inter_cnn_3d, axis=0)
        mean_hd_inter_cnn = np.mean(hd_scores_inter_cnn, axis=0)
        mean_hd95_inter_cnn = np.mean(hd95_scores_inter_cnn, axis=0)
    else:
        mean_inter_cnn_2d = np.mean(np.mean(class_scores_inter_cnn_2d, axis=0), axis=0)
        mean_inter_cnn_3d = np.mean(np.mean(class_scores_inter_cnn_3d, axis=0), axis=0)
        mean_hd_inter_cnn = np.mean(np.mean(hd_scores_inter_cnn, axis=0), axis=0)
        mean_hd95_inter_cnn = np.mean(np.mean(hd95_scores_inter_cnn, axis=0), axis=0)

    print("\n2D autoCNN/interCNN accuracy: {} / {} \tSignificant autoCNN/interCNN accuracy: {:.4f} / {:.4f}"
          .format(np.around(mean_auto_cnn_2d, 4), np.around(mean_inter_cnn_2d, 4),
                  np.mean(mean_auto_cnn_2d[1:]), np.mean(mean_inter_cnn_2d[1:])))
    print("3D autoCNN/interCNN accuracy: {} / {} \tSignificant autoCNN/interCNN accuracy: {:.4f} / {:.4f} \n"
          .format(np.around(mean_auto_cnn_3d, 4), np.around(mean_inter_cnn_3d, 4),
                  np.mean(mean_auto_cnn_3d[1:]), np.mean(mean_inter_cnn_3d[1:])))

    print("\nHD autoCNN/interCNN accuracy: {} / {} \tSignificant autoCNN/interCNN accuracy: {:.4f} / {:.4f}"
          .format(np.around(mean_hd_auto_cnn, 4), np.around(mean_hd_inter_cnn, 4),
                  np.mean(mean_hd_auto_cnn[1:]), np.mean(mean_hd_inter_cnn[1:])))
    print("HD95 autoCNN/interCNN accuracy: {} / {} \tSignificant autoCNN/interCNN accuracy: {:.4f} / {:.4f} \n"
          .format(np.around(mean_hd95_auto_cnn, 4), np.around(mean_hd95_inter_cnn, 4),
                  np.mean(mean_hd95_auto_cnn[1:]), np.mean(mean_hd95_inter_cnn[1:])))

    np.save('autocnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_2d_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
                    num_filters_auto_cnn, learning_rate, bool_data_augmentation), class_scores_auto_cnn_2d)
    np.save('autocnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_3d_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
                    num_filters_auto_cnn, learning_rate, bool_data_augmentation), class_scores_auto_cnn_3d)
    np.save('intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_2d_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                    num_filters_inter_cnn, learning_rate, bool_data_augmentation), class_scores_inter_cnn_2d)
    np.save('intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_3d_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                    num_filters_inter_cnn, learning_rate, bool_data_augmentation), class_scores_inter_cnn_3d)

    np.save('autocnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_hd_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
                    num_filters_auto_cnn, learning_rate, bool_data_augmentation), hd_scores_auto_cnn)
    np.save('autocnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_hd95_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
                    num_filters_auto_cnn, learning_rate, bool_data_augmentation), hd95_scores_auto_cnn)
    np.save('intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_hd_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
                    num_filters_auto_cnn, learning_rate, bool_data_augmentation), hd_scores_inter_cnn)
    np.save('intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_hd95_test_scores.npy'
            .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_auto_cnn,
                    num_filters_auto_cnn, learning_rate, bool_data_augmentation), hd95_scores_inter_cnn)
    print_time(start=start_timer)
    print("#" * 46 + "\n" + "#" * 20 + " DONE " + "#" * 20 + "\n" + "#" * 46)


def add_bool_arg(bool_parser, names, destination, helps, default=False):
    group = bool_parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + names[0], dest=destination, action='store_true', help=helps[0])
    group.add_argument('--' + names[1], dest=destination, action='store_false', help=helps[1])
    bool_parser.set_defaults(**{destination: default})


def create_parser():
    # %% Training settings
    parser = argparse.ArgumentParser(prog='brats_prostate_gustavs_splits',
                                     description='Prostate / BraTS training with Gustav´s data splits',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=True)

    add_bool_arg(parser, names=('use_prostate_data', 'use_brats_data'), destination='bool_prostate_data',
                 helps=('use the NCI-ISBI Prostate dataset', 'use the BraTS 2015 dataset'))

    parser.add_argument('--inter_mode', type=str, metavar='str', default='2d', choices=['2d', '3d'],
                        help='Select the mode of scribbles; %(choices)s')

    parser.add_argument('--os', type=str, metavar='str', default='linux', choices=['linux', 'windows'],
                        help='Select the OS; %(choices)s')

    parser.add_argument('--enable_binary_testing', action='store_true',
                        help='Enable binary classification')

    parser.add_argument('--number_of_scribbles', type=int, metavar='N', default=1,
                        help='Select the number of scribbles to generate in each iteration')

    parser.add_argument('--size_of_scribble', type=int, metavar='N', default=4,
                        help='Select the size (sigma) of each scribble')

    parser.add_argument('--place_in_biggest_vol', action='store_true',
                        help='Place the scribbles in the biggest connected component')

    return parser


if __name__ == '__main__':
    arg_parser = create_parser()
    args = arg_parser.parse_args()
    print(args)

    if args.enable_binary_testing:
        load_model_folder_auto = \
            '/scratch_net/giggo/shlynur/msc/polybox/M.Sc.2019/code/shlynur_unet_testing/3d_results' \
            '/auto_cnn_models/binary/gustav_split_'
    else:
        load_model_folder_auto = \
            '/scratch_net/giggo/shlynur/msc/polybox/M.Sc.2019/code/shlynur_unet_testing/3d_results' \
            '/auto_cnn_models/non_binary/gustav_split_'

    load_model_file_auto = 'auto_cnn_3D_model_gustav_epochs_500_filters_32_lr_0.0003_dataug_True.pt'

    for g in range(1, 4 if args.enable_binary_testing else 6):
        brats_prostate_gustavs_splits(
            seed=48, bool_prostate_data=args.bool_prostate_data, model_type='gustav', num_filters_auto_cnn=32,
            num_filters_inter_cnn=32, num_epochs_auto_cnn=500, num_epochs_inter_cnn=50, batch_size=1,
            learning_rate=3e-4, lr_wd=1e-4, bool_load_previous_checkpoint=(False, False),
            load_model_folder=[load_model_folder_auto + str(g), ''], load_model_file=[load_model_file_auto, ''],
            loss_weight=None, max_iterations_train=10, max_iterations_test=20, bool_auto_cnn_train=False,
            bool_inter_cnn_train=False, bool_save_results=True, bool_data_augmentation=True,
            inter_number_of_slices='all', inter_scribble_mode=args.inter_mode, operating_system=args.os,
            gustav_split=g, bool_binary_testing=args.enable_binary_testing,
            bool_loss_summation=False, number_of_scribbles=args.number_of_scribbles,
            size_of_scribble=args.size_of_scribble, bool_best_placement=args.place_in_biggest_vol)
