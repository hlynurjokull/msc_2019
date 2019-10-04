# import os
import gc
import logging
# import string
import h5py

import numpy as np
# from skimage import transform
import nibabel as nib
# from utils.shlynur_data_augmentation import min_max_normalization


def h5print_r(item, leading=''):
    # Helper function for printing the structure of a .h5 file
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5print_r(item[key], leading + '  ')


# Print structure of a `.h5` file
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print("#" * 60 + "\n\t\t\tTHE STRUCTURE OF THE HDF5 FILE\n" + "#" * 60)
        print("Filename: {}".format(filename))
        h5print_r(h, '  ')


def load_nii(img_path):
    """
    Shortcut to load a nifti file
    """

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


# def test_train_val_split(patient_id, data_splits):
#     # if patient_id % 10 == 0:
#     #     return 'test'
#     # elif patient_id % 10 == 1:
#     #     return 'validation'
#     # else:
#     #     return 'train'
#     #
#     # if patient_id % 10 == 0:
#     #     return 'test'
#     # elif patient_id % 10 == 1:
#     #     return 'autocnn_val'
#     # elif patient_id % 10 == 2:
#     #     return 'intercnn_val'
#     # else:
#     #     return 'autocnn_train'
#     data_splits_2 = tuple(i/100*210 for i in data_splits)
#     new_data_splits = [data_splits_2[0]]
#     for i in range(1, len(data_splits_2)):
#         new_data_splits.append(data_splits_2[i] + new_data_splits[i-1])
#
#     if patient_id < new_data_splits[0]:
#         return 'autocnn_train'
#     elif patient_id < new_data_splits[1]:
#         return 'autocnn_val'
#     elif patient_id < new_data_splits[2]:
#         return 'intercnn_val'
#     else:
#         return 'test'


def crop_volume_all_dim(image, mask=None):
    """
    Strip away the zeros on the edges of the three dimensions of the image
    Idea: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
    """
    coordinates = np.argwhere(image > 0)
    _, z0, x0, y0 = coordinates.min(axis=0)
    _, z1, x1, y1 = coordinates.max(axis=0) + 1

    image = image[:, z0:z1, x0:x1, y0:y1]
    if mask is None:
        return image
    else:
        return image, mask[z0:z1, x0:x1, y0:y1]


def pad_slice_to_size(image, mask, target_size):
    """
    Pad the image and mask to the desired target size.
    """

    difference = np.subtract(target_size, image.shape[1:])
    difference_div_2 = difference // 2
    difference_difference = difference - difference_div_2

    pad_width_image = ((0, 0),
                       (difference_div_2[0], difference_difference[0]),
                       (difference_div_2[1], difference_difference[1]),
                       (difference_div_2[2], difference_difference[2]))
    pad_width_mask = ((difference_div_2[0], difference_difference[0]),
                      (difference_div_2[1], difference_difference[1]),
                      (difference_div_2[2], difference_difference[2]))

    output_image = np.pad(array=image, pad_width=pad_width_image, mode='constant')
    output_mask = np.pad(array=mask, pad_width=pad_width_mask, mode='constant')

    return output_image, output_mask


def _write_range_to_hdf5(hdf5_data, train_val_test, img_list, mask_list,  # pids_list,
                         counter_from, counter_to):
    """
    Helper function to write a range of data to the hdf5 datasets
    """

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))
    img_arr = np.asarray(img_list[train_val_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_val_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_val_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_val_test][counter_from:counter_to, ...] = mask_arr
    # hdf5_data['pids_%s' % train_val_test][counter_from:counter_to, ...] = pids_list[train_val_test]


def _release_tmp_memory(img_list, mask_list,  # pids_list,
                        train_val_test):
    """
    Helper function to reset the tmp lists and free the memory
    """

    img_list[train_val_test].clear()
    mask_list[train_val_test].clear()
    # pids_list[train_val_test].clear()
    gc.collect()


def load_brats_data_3d(input_folder, output_file, max_buffer=5, size=(155, 240, 240), gustav_split=1,
                       input_channels=4):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    print("CREATING HDF5 FILE FROM GUSTAV'S SPLIT # {}".format(gustav_split))

    hdf5_file = h5py.File(output_file, "w")
    file_list = {'autocnn_train': [], 'autocnn_val': [], 'intercnn_train': [], 'intercnn_val': [], 'test': []}
    group_list = {'autocnn', 'intercnn'}

    logging.info('Counting files and parsing meta data...')

    gustav_data_split = np.load('/scratch_net/giggo/shlynur/msc/polybox/M.Sc.2019/code/shlynur_unet_testing/'
                                'gustav_code/BRATS_data-split/data_split_{}.npy'.format(gustav_split))

    # Split the data into training/validating/testing.
    for i in range(200):
        number = gustav_data_split[i]
        if i < 100:
            file_list['autocnn_train'].append(number)
        elif i < 150:
            file_list['autocnn_val'].append(number)
        elif i < 155:
            file_list['intercnn_val'].append(number)
        else:
            file_list['test'].append(number)

    file_list['intercnn_train'] = file_list['autocnn_train'] + file_list['autocnn_val']

    n_images = {'autocnn_train': [], 'autocnn_val': [], 'intercnn_train': [], 'intercnn_val': [], 'test': []}
    for i in file_list:
        n_images[i] = len(file_list[i])
    print(n_images)
    n_autocnn_train, n_autocnn_val, n_intercnn_train, n_intercnn_val, n_test = [len(file_list[i]) for i in file_list]

    data = {}
    for gg in group_list:
        hdf5_file.create_group("/%s" % gg)
        for tt, num_points in zip(['%s_train' % gg, '%s_val' % gg],
                                  [n_images['%s_train' % gg], n_images['%s_val' % gg]]):
            print(tt, num_points)
            data['images_%s' % tt] = hdf5_file.create_dataset("/%s/images_%s" % (gg, tt),
                                                              [num_points] + [input_channels] + list(size),
                                                              dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("/%s/masks_%s" % (gg, tt), [num_points] + list(size),
                                                             dtype=np.float32)
    hdf5_file.create_group("/test")
    data['images_test'] = hdf5_file.create_dataset("/test/images_test", [n_test] + [input_channels] + list(size),
                                                   dtype=np.float32)
    data['masks_test'] = hdf5_file.create_dataset("/test/masks_test", [n_test] + list(size), dtype=np.float32)

    h5print(output_file)

    img_list = {'autocnn_train': [], 'autocnn_val': [], 'intercnn_train': [], 'intercnn_val': [], 'test': []}
    mask_list = {'autocnn_train': [], 'autocnn_val': [], 'intercnn_train': [], 'intercnn_val': [], 'test': []}

    logging.info('Parsing image files')

    for train_val_test in ['autocnn_train', 'autocnn_val', 'intercnn_train', 'intercnn_val', 'test']:
        write_buffer = 0
        counter_from = 0
        for number in file_list[train_val_test]:
            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s \t%s' % (train_val_test, number))

            seg_path = 'preprocessedtciafrom' + str(number) + 'to' + str(number + 1) + '_annotations.npy'
            flair_path = 'preprocessedtciafrom' + str(number) + 'to' + str(number + 1) + '_images_flr_n4_norm2.npy'
            t1_path = 'preprocessedtciafrom' + str(number) + 'to' + str(number + 1) + '_images_t1_n4_norm2.npy'
            t1c_path = 'preprocessedtciafrom' + str(number) + 'to' + str(number + 1) + '_images_t1c_n4_norm2.npy'
            t2_path = 'preprocessedtciafrom' + str(number) + 'to' + str(number + 1) + '_images_t2_n4_norm2.npy'

            flair_file = np.load(input_folder + flair_path)
            t1_file = np.load(input_folder + t1_path)
            t1c_file = np.load(input_folder + t1c_path)
            t2_file = np.load(input_folder + t2_path)
            mask_dat = np.load(input_folder + seg_path)

            img_dat = np.stack((flair_file, t1_file, t1c_file, t2_file), 0)
            img, mask = crop_volume_all_dim(img_dat, mask_dat)
            img, mask = pad_slice_to_size(image=img, mask=mask, target_size=size)

            img_list[train_val_test].append(img)
            mask_list[train_val_test].append(mask)

            write_buffer += 1

            if write_buffer >= max_buffer:
                counter_to = counter_from + write_buffer
                _write_range_to_hdf5(hdf5_data=data, train_val_test=train_val_test, img_list=img_list,
                                     mask_list=mask_list,  # pids_list=pids_list,
                                     counter_from=counter_from,
                                     counter_to=counter_to)
                _release_tmp_memory(img_list=img_list, mask_list=mask_list,  # pids_list=pids_list,
                                    train_val_test=train_val_test)

                counter_from = counter_to
                write_buffer = 0

        logging.info('Writing remaining data.')
        counter_to = counter_from + write_buffer

        if len(file_list[train_val_test]) > 0:
            _write_range_to_hdf5(hdf5_data=data, train_val_test=train_val_test, img_list=img_list, mask_list=mask_list,
                                 # pids_list=pids_list,
                                 counter_from=counter_from, counter_to=counter_to)
        _release_tmp_memory(img_list=img_list, mask_list=mask_list,  # pids_list=pids_list,
                            train_val_test=train_val_test)

    # After the loop:
    hdf5_file.close()


if __name__ == '__main__':
    # Create the hdf5 files that contain the 2015 BraTS data. 5 data splits => 5 hdf5 files.

    max_write_buffer = 50
    volume_size = (160, 192, 160)
    volume_size_string = '_'.join(str(i) for i in volume_size)
    data_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Brats_preprocessed_silvan/preprocessed/'
    output_folder = '/scratch_net/giggo/shlynur/msc/'

    for gustav_data in range(1, 6):
        output_hdf5_file = output_folder + 'shlynur_brats_data_3D_size_{}_gustavs_split_{}.hdf5'\
            .format(volume_size_string, gustav_data)
        load_brats_data_3d(input_folder=data_folder, output_file=output_hdf5_file, max_buffer=max_write_buffer,
                           size=volume_size, input_channels=4, gustav_split=gustav_data)
