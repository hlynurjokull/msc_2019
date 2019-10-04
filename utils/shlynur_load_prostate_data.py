import os
import numpy as np
import pydicom as dicom
import nrrd


def load_prostate_images_and_labels_2d(train_ids_list, data_path, seg_path):
    """
    Load the prostate dataset in a 2D setting.
    :param train_ids_list: ID list of the subjects.
    :param data_path: location of the images.
    :param seg_path: location of the segmentation masks.
    :return: images and segmentation masks.
    """
    img_final, seg_final = [], []
    count = 0
    for study_id in train_ids_list:
        path_dicom = (data_path + str(study_id))
        # Create an empty list for the DICOM files.
        dcm_files = []
        for dirName, subdirList, fileList in os.walk(path_dicom):
            fileList.sort()
            for filename in fileList:
                if ".dcm" in filename.lower():  # If the selected file is a DICOM.
                    dcm_files.append(os.path.join(dirName, filename))
        ref_file = dicom.read_file(dcm_files[0])

        const_pixel_dims = (int(ref_file.Rows), int(ref_file.Columns), len(dcm_files))

        img = np.zeros(const_pixel_dims, dtype=ref_file.pixel_array.dtype)

        for dcm_filename in dcm_files:
            # Read the file.
            ds = dicom.read_file(dcm_filename)
            # Store the raw image data.
            img[:, :, ds.InstanceNumber-1] = ds.pixel_array

        path_segmentation = (seg_path + str(study_id) + ".nrrd")
        seg, options = nrrd.read(path_segmentation)

        # Fix swapped axes.
        seg = np.swapaxes(seg, 0, 1)

        if count == 0:
            img_final = img
            seg_final = seg
        else:
            img_final = np.concatenate((img_final, img), axis=2)
            seg_final = np.concatenate((seg_final, seg), axis=2)
        count = count + 1
    return img_final, seg_final


def load_prostate_images_and_labels_3d(train_ids_list, data_path, seg_path):
    """
    Load the prostate dataset in a 3D setting. Sub-sampling each volume to 15 slices.
    :param train_ids_list: ID list of the subjects.
    :param data_path: location of the images.
    :param seg_path: location of the segmentation masks.
    :return: images and segmentation masks.
    """
    img_final, seg_final = [], []
    for study_id in train_ids_list:
        path_dicom = (data_path + str(study_id))
        # Create an empty list for the DICOM files.
        dcm_files = []
        for dirName, subdirList, fileList in os.walk(path_dicom):
            fileList.sort()
            for filename in fileList:
                if ".dcm" in filename.lower():  # If the selected file is a DICOM.
                    dcm_files.append(os.path.join(dirName, filename))
        ref_file = dicom.read_file(dcm_files[0])

        const_pixel_dims = (len(dcm_files), int(ref_file.Rows), int(ref_file.Columns))

        img = np.zeros(const_pixel_dims, dtype=ref_file.pixel_array.dtype)

        for dcm_filename in dcm_files:
            # Read the file.
            ds = dicom.read_file(dcm_filename)
            # Store the raw image data.
            img[ds.InstanceNumber - 1, :, :] = ds.pixel_array

        path_segmentation = (seg_path + str(study_id) + ".nrrd")
        seg, options = nrrd.read(path_segmentation)

        # Fix swapped axes.
        seg = np.swapaxes(seg, 0, 2)

        # 15 slice sub-sampling.
        margin = img.shape[0] - 15
        if margin > 0:
            for x in range(margin + 1):
                new_img = img[x: x + 15, :, :]
                new_seg = seg[x: x + 15, :, :]
                img_final.append(new_img)
                seg_final.append(new_seg)
        else:
            img_final.append(img)
            seg_final.append(seg)

    return img_final, seg_final


def load_prostate_images_and_labels_3d_test(train_ids_list, data_path, seg_path):
    """
    Load the prostate dataset in a 3D setting. Without any sub-sampling.
    :param train_ids_list: ID list of the subjects.
    :param data_path: location of the images.
    :param seg_path: location of the segmentation masks.
    :return: images and segmentation masks.
    """
    img_final, seg_final = [], []
    for study_id in train_ids_list:
        path_dicom = (data_path + str(study_id))
        # Create an empty list for the DICOM files.
        dcm_files = []
        for dirName, subdirList, fileList in os.walk(path_dicom):
            fileList.sort()
            for filename in fileList:
                if ".dcm" in filename.lower():  # If the selected file is a DICOM.
                    dcm_files.append(os.path.join(dirName, filename))
        ref_file = dicom.read_file(dcm_files[0])

        const_pixel_dims = (len(dcm_files), int(ref_file.Rows), int(ref_file.Columns))

        img = np.zeros(const_pixel_dims, dtype=ref_file.pixel_array.dtype)

        for dcm_filename in dcm_files:
            # Read the file.
            ds = dicom.read_file(dcm_filename)
            # Store the raw image data.
            img[ds.InstanceNumber - 1, :, :] = ds.pixel_array

        path_segmentation = (seg_path + str(study_id) + ".nrrd")
        seg, options = nrrd.read(path_segmentation)

        # Fix swapped axes.
        seg = np.swapaxes(seg, 0, 2)

        img_final.append(img)
        seg_final.append(seg)
    return img_final, seg_final
