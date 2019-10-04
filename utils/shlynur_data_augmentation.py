import random
import numpy as np
import scipy.ndimage.interpolation as interpolation
# import bruegger_code.brats.dataProcessing.utils as utils
# from matplotlib import pyplot as plt
# import time
import cv2
# import os
# import nibabel as nib
# from skimage import measure
import time

# random_factor = 0.8705505632961241  # select this one for a 50% chance of 0/5 data augmentation techniques applied.
random_factor = 0.5


def min_max_normalization(image, method='z_score'):
    """
    Normalize the image.
    :param image: input image.
    :param method: The desired normalization method. Currently: 'zero_to_one' or 'z_score'.
    :return: normalized_image.
    """
    if method == 'zero_to_one':
        image_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)))
    elif method == 'z_score':
        image_normalized = (image - np.mean(image)) / np.std(image)
    else:
        raise ValueError('The normalization method must be either "z_score" or "zero_to_one".')

    return image_normalized


def aug_flip(image, label, axis, bool_prostate_data=True, randomness=True):
    """
    Flip the image and label.
    :param image: input image.
    :param label: input label.
    :param axis: 0 for a horizontal flip, 1 for a vertical flip.
    :param bool_prostate_data: select a dataset, True for Prostate, False for BraTS
    :param randomness: if False, disregard the random factor and apply the augmentation.
    :return: flipped image and label.
    """
    random.seed(time.clock())
    if randomness:
        random_number = random.random()
    else:
        random_number = 1
    if random_number < random_factor:
        return image, label

    if bool_prostate_data:
        image_flip = np.flip(image, axis=axis)
    else:
        image_flip = np.flip(image, axis=(axis + 1))
    label_flip = np.flip(label, axis=axis)

    image_flip = np.ascontiguousarray(image_flip)
    label_flip = np.ascontiguousarray(label_flip)

    return image_flip, label_flip


def aug_rotate(image, label, angle, axes, bool_prostate_data=True, randomness=True):
    """
    Rotate the image and label.
    :param image: input image.
    :param label: input label.
    :param angle: the degree of rotation is a random integer in [-angle, +angle].
    :param axes: the plane of rotation is defined by the axes.
    :param bool_prostate_data: select a dataset, True for Prostate, False for BraTS
    :param randomness: if False, disregard the random factor and apply the augmentation.
    :return: rotated image and label.
    """
    random.seed(time.clock())
    if randomness:
        random_number = random.random()
    else:
        random_number = 1
    if random_number < random_factor:
        return image, label

    degree = random.randint(-angle, angle)

    if bool_prostate_data:
        image_rot = interpolation.rotate(input=image, angle=degree, axes=axes, reshape=False, order=3, mode='reflect')
    else:
        new_axes = tuple(x + y for x, y in zip(axes, (1, 1)))  # adding 1 to each value in the tuple 'axis'
        image_rot = interpolation.rotate(input=image, angle=degree, axes=new_axes, reshape=False, order=3,
                                         mode='reflect')
    label_rot = interpolation.rotate(input=label, angle=degree, axes=axes, reshape=False, order=0, mode='reflect')

    return image_rot, label_rot


def aug_zoom(image, label, scale_factor, dimensions=2, bool_prostate_data=True, randomness=True):
    """
    Crop the images.
    :param image: input image.
    :param label: input label.
    :param scale_factor: size of the zoom.
    :param dimensions: '2' or '3' for 2D or 3D
    :param bool_prostate_data: select a dataset, True for Prostate, False for BraTS
    :param randomness: if False, disregard the random factor and apply the augmentation.
    :return: zoomed image and label.
    """
    random.seed(time.clock())
    if randomness:
        random_number = random.random()
    else:
        random_number = 1
    if random_number < random_factor:
        return image, label

    crop_size = np.random.uniform(1 / scale_factor, 1 * scale_factor)
    image_size = image.shape[-2:]

    if dimensions == 2:
        # Zoom into the image.
        image_zoom_slice = interpolation.zoom(image, zoom=crop_size, order=1)
        label_zoom_slice = interpolation.zoom(label, zoom=crop_size, order=0)

        # Crop the image.
        x, y = image_zoom_slice.shape
        start_x = round(x / 2) - round(image_size[0] / 2)
        start_y = round(y / 2) - round(image_size[1] / 2)
        end_x = start_x + image_size[0]
        end_y = start_y + image_size[1]

        image_zoomed = image_zoom_slice[start_x:end_x, start_y:end_y]
        label_zoomed = label_zoom_slice[start_x:end_x, start_y:end_y]

    else:
        image_zoomed, label_zoomed = np.zeros(image.shape), np.zeros(label.shape)

        if bool_prostate_data:
            for i in range(image.shape[0]):
                # Zoom into each slice.
                image_zoom_slice = interpolation.zoom(image[i, :], zoom=crop_size, order=1)
                label_zoom_slice = interpolation.zoom(label[i, :], zoom=crop_size, order=0)

                x, y = image_zoom_slice.shape

                if crop_size >= 1:
                    # Crop each slice.
                    start_x = (x - image_size[0]) // 2
                    start_y = (y - image_size[1]) // 2
                    end_x = start_x + image_size[0]
                    end_y = start_y + image_size[1]

                    image_zoomed[i, :] = image_zoom_slice[start_x:end_x, start_y:end_y]
                    label_zoomed[i, :] = label_zoom_slice[start_x:end_x, start_y:end_y]
                else:
                    # zero-pad each slice.
                    new_image = np.zeros(image_size)
                    new_label = np.zeros(image_size)

                    start_x = (image_size[0] - x) // 2
                    start_y = (image_size[1] - y) // 2
                    end_x = start_x + x
                    end_y = start_y + y

                    new_image[start_x:end_x, start_y:end_y] = image_zoom_slice
                    new_label[start_x:end_x, start_y:end_y] = label_zoom_slice

                    image_zoomed[i, :] = new_image
                    label_zoomed[i, :] = new_label

        else:
            for i in range(image.shape[1]):
                for j in range(image.shape[0]):
                    # Zoom into each slice.
                    image_zoom_slice = interpolation.zoom(image[j, i, :], zoom=crop_size, order=1)
                    # label_zoom_slice = interpolation.zoom(label[i, :], zoom=crop_size, order=0)

                    x, y = image_zoom_slice.shape

                    if crop_size >= 1:
                        # Crop each slice.
                        start_x = (x - image_size[0]) // 2
                        start_y = (y - image_size[1]) // 2
                        end_x = start_x + image_size[0]
                        end_y = start_y + image_size[1]

                        image_zoomed[j, i, :] = image_zoom_slice[start_x:end_x, start_y:end_y]
                        # label_zoomed.append(label_zoom_slice[start_x:end_x, start_y:end_y])
                    else:
                        # zero-pad each slice.
                        new_image = np.zeros(image_size)
                        # new_label = np.zeros((image_size, image_size))

                        start_x = (image_size[0] - x) // 2
                        start_y = (image_size[1] - y) // 2
                        end_x = start_x + x
                        end_y = start_y + y

                        new_image[start_x:end_x, start_y:end_y] = image_zoom_slice
                        # new_label[start_x:end_x, start_y:end_y] = label_zoom_slice

                        image_zoomed[j, i, :] = new_image
                        # label_zoomed.append(new_label)

                label_zoom_slice = interpolation.zoom(label[i, :], zoom=crop_size, order=0)

                x, y = label_zoom_slice.shape

                if crop_size >= 1:
                    # Crop each slice.
                    start_x = (x - image_size[0]) // 2
                    start_y = (y - image_size[1]) // 2
                    end_x = start_x + image_size[0]
                    end_y = start_y + image_size[1]

                    label_zoomed[i, :] = label_zoom_slice[start_x:end_x, start_y:end_y]
                else:
                    # zero-pad each slice.
                    new_label = np.zeros(image_size)

                    start_x = (image_size[0] - x) // 2
                    start_y = (image_size[1] - y) // 2
                    end_x = start_x + x
                    end_y = start_y + y

                    new_label[start_x:end_x, start_y:end_y] = label_zoom_slice

                    label_zoomed[i, :] = new_label

    return image_zoomed, label_zoomed


def aug_elastic_deform(image, label, mu, sigma, interpolation_method, bool_prostate_data=True, randomness=True):
    """
    Apply elastic deformations to the image and label.
    :param image: input image.
    :param label: input label.
    :param mu: mean of the gaussian distribution.
    :param sigma: standard deviation of the gaussian distribution.
    :param interpolation_method:
    :param bool_prostate_data: select a dataset, True for Prostate, False for BraTS.
    :param randomness: if False, disregard the random factor and apply the augmentation.
    :return: elastically deformed image and label.
    """
    random.seed(time.clock())
    if randomness:
        random_number = random.random()
    else:
        random_number = 1
    if random_number < random_factor:
        return image, label

    if bool_prostate_data:
        z_size, x_size, y_size = image.shape
        channels = 1
    else:
        channels, z_size, x_size, y_size = image.shape

    dx = np.random.normal(mu, sigma, 9)
    dx_mat = np.reshape(dx, (3, 3))
    dx_img = cv2.resize(src=dx_mat, dsize=(y_size, x_size), interpolation=cv2.INTER_CUBIC)

    dy = np.random.normal(mu, sigma, 9)
    dy_mat = np.reshape(dy, (3, 3))
    dy_img = cv2.resize(src=dy_mat, dsize=(y_size, x_size), interpolation=cv2.INTER_CUBIC)

    image_deform, label_deform = np.zeros(image.shape), np.zeros(label.shape)

    if bool_prostate_data:
        for z in range(z_size):
            image_deform[z, :] = dense_image_warp(image[z, :], dx_img, dy_img)
            label_deform[z, :] = dense_image_warp(label[z, :], dx_img, dy_img, interpolation_method)
    else:
        for z in range(z_size):
            for c in range(channels):
                image_deform[c, z, :] = dense_image_warp(image[c, z, :], dx_img, dy_img)
            label_deform[z, :] = dense_image_warp(label[z, :], dx_img, dy_img, interpolation_method)

    return image_deform, label_deform


def aug_intensity_shift(image, max_intensity_shift, bool_prostate_data=True, randomness=True):
    """
    Apply intensity shift to each channel in the image.
    :param image: input image.
    :param bool_prostate_data: select a dataset, True for Prostate, False for BraTS
    :param max_intensity_shift: the maximum intensity shift amount.
    :param randomness: if False, disregard the random factor and apply the augmentation.
    :return: intensity shifted image.
    """
    random.seed(time.clock())
    if randomness:
        random_number = random.random()
    else:
        random_number = 1
    if random_number < random_factor:
        return image

    if bool_prostate_data:
        image_shifted = image + np.random.uniform(-max_intensity_shift, max_intensity_shift)
    else:
        image_shifted = np.zeros(image.shape)
        for i in range(image.shape[0]):
            image_shifted[i, :] = image[i, :] + np.random.uniform(-max_intensity_shift, max_intensity_shift)

    return image_shifted


# # Image shape: (20, 320, 320)
# def brueggger_augment_3d_image(
#         image, label, nn_aug, default_label_values=0, do_flip=True, axes_flip_h=2, axes_flip_v=1, do_rotate=True,
#         rot_degrees=20, axes_rot=(1, 2), do_scale=True, scale_factor=1.1, do_elastic_aug=True, mu=0, sigma=10,
#         do_intensity_shift=True, max_intensity_shift=0.1):
#     """
#     Function for augmentation of a 3D image. It will transform the image and corresponding labels
#     by a number of optional transformations.
#     :param image: A numpy array of shape [X, Y, Z, nChannels]
#     :param label: A numpy array containing a corresponding label mask
#     :param nn_aug:
#     :param default_label_values:
#     :param do_flip: Perform random flips with a 50% chance in the left right direction.
#     :param axes_flip_h:
#     :param axes_flip_v:
#     :param do_rotate: Rotate the input images by a random angle between -15 and 15 degrees.
#     :param rot_degrees:
#     :param axes_rot:
#     :param do_scale: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
#                         back to the original size.
#     :param scale_factor
#     :param do_elastic_aug:
#     :param mu:
#     :param sigma:
#     :param do_intensity_shift:
#     :param max_intensity_shift
#     :return: Transformed images and masks.
#     """
#
#     # start_time = time.time()
#     x_size = image.shape[1]
#     y_size = image.shape[2]
#     z_size = image.shape[0]
#     default_per_channel = image[0, 0, 0, :]
#
#     new_image = image
#     new_label = label
#
#     if nn_aug:
#         interpolation_method = cv2.INTER_NEAREST
#     else:
#         interpolation_method = cv2.INTER_LINEAR
#
#     # visualize augmentation
#     # halfOffset = z_size // 6
#     # sliceIndices = [halfOffset, 3*halfOffset, 5*halfOffset]
#     # for i in range(len(sliceIndices)):
#     #     visualizeSlice(image[:, :, sliceIndices[i], 0])
#     #     visualizeSlice(label[:, :, sliceIndices[i], 0])
#
#     # FLIP UD/LR
#     new_image, new_label = aug_flip(image=new_image, label=new_label, axis=axes_flip_h)
#     # new_image, new_label = aug_flip(image=new_image, label=new_label, axis=axes_flip_v)
#
#     # ROTATE
#     new_image, new_label = aug_rotate(image=new_image, label=new_label, angle=rot_degrees, axes=axes_rot)
#     # if do_rotate:
#     #     random_angle = np.random.uniform(-rot_degrees, rot_degrees)
#     #     for z in range(z_size):
#     #         # image[:, :, z, :] = utils.rotate_image(image[:, :, z, :], random_angle)
#     #         # label[:, :, z, :] = utils.rotate_image(label[:, :, z, :], random_angle, interpolation_method)
#     #         image[z, :, :, :] = rotate_image(image[z, :, :, :], random_angle)
#     #         label[z, :, :, :] = rotate_image(label[z, :, :, :], random_angle, interpolation_method)
#
#     # RANDOM SCALE
#     new_image, new_label = aug_zoom(image=new_image, label=new_label, scale_factor=scale_factor, dimensions=3)
#     # if do_scale:
#     #     scale = np.random.uniform(1 / scale_factor, 1 * scale_factor)
#     #     for z in range(z_size):
#     #         scaled_size = [round(x_size*scale), round(y_size*scale)]
#     #         img_scaled = resize_image(image[z, :, :, :], scaled_size)
#     #         lbl_scaled = resize_image(image[z, :, :, :], scaled_size, interpolation_method)
#     #         if scale < 1:
#     #             image[z, :, :, :] = pad_to_size(img_scaled, [x_size, y_size], default_per_channel)
#     #             label[z, :, :, :] = pad_to_size(lbl_scaled, [x_size, y_size], default_label_values)
#     #         else:
#     #             image[z, :, :, :] = crop_to_size(img_scaled, [x_size, y_size])
#     #             label[z, :, :, :] = crop_to_size(lbl_scaled, [x_size, y_size])
#
#     # RANDOM ELASTIC DEFORMATIONS (like in U-NET)
#     new_image, new_label = aug_elastic_deform(image=new_image, label=new_label, mu=mu, sigma=sigma,
#                                               interpolation_method=interpolation_method)
#     # if do_elastic_aug:
#     #
#     #     mu = 0
#     #
#     #     dx = np.random.normal(mu, sigma, 9)
#     #     dx_mat = np.reshape(dx, (3, 3))
#     #     dx_img = resize_image(dx_mat, (x_size, y_size), interp=cv2.INTER_CUBIC)
#     #
#     #     dy = np.random.normal(mu, sigma, 9)
#     #     dy_mat = np.reshape(dy, (3, 3))
#     #     dy_img = resize_image(dy_mat, (x_size, y_size), interp=cv2.INTER_CUBIC)
#     #
#     #     for z in range(z_size):
#     #         image[:, :, z, :] = dense_image_warp(image[:, :, z, :], dx_img, dy_img)
#     #         label[:, :, z, :] = dense_image_warp(label[:, :, z, :], dx_img, dy_img, interpolation_method)
#
#     # RANDOM INTENSITY SHIFT
#     if do_intensity_shift:
#         for i in range(1):  # number of channels
#             image[:, :, :, i] = image[:, :, :, i] + np.random.uniform(-max_intensity_shift, max_intensity_shift)
#             # assumes unit std derivation
#
#     # RANDOM FLIP
#     # if do_flip:
#     #     for i in range(3):
#     #         if np.random.random() < 0.5:
#     #             image = np.flip(image, axis=i)
#     #             label = np.flip(label, axis=i)
#
#     # log augmentation time
#     # print("Augmentation took {}s".format(time.time() - start_time))
#
#     # visualize augmentation
#     # halfOffset = z_size // 6
#     # sliceIndices = [halfOffset, 3*halfOffset, 5*halfOffset]
#     # for i in range(len(sliceIndices)):
#     #     visualizeSlice(image[:, :, sliceIndices[i], 0])
#     #     visualizeSlice(label[:, :, sliceIndices[i], 4])
#
#     return new_image, new_label
#     # return image.copy(), label.copy()  # pytorch cannot handle negative stride in view


# def visualize_slice(slice_no):
#     plt.imshow(slice_no, interpolation='nearest')
#     plt.show()


# def crop_to_size(image, target_size):
#     offset_x = (image.shape[0] - target_size[0]) // 2
#     end_x = offset_x + target_size[0]
#     offset_y = (image.shape[1] - target_size[1]) // 2
#     end_y = offset_y + target_size[1]
#     return image[offset_x:end_x, offset_y:end_y, :]


# def pad_to_size(image, target_size, background_color):
#     offset_x = (target_size[0] - image.shape[0]) // 2
#     end_x = offset_x + image.shape[0]
#     offset_y = (target_size[1] - image.shape[1]) // 2
#     end_y = offset_y + image.shape[1]
#     target_size.append(image.shape[2])  # add channels to shape
#     padded_img = np.ones(target_size, dtype=np.float32) * background_color
#     padded_img[offset_x:end_x, offset_y:end_y, :] = image
#     return padded_img
#
#
# def rotate_image(img, angle, interp=cv2.INTER_LINEAR):
#     rows, cols = img.shape[:2]
#     rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     out = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp, borderMode=cv2.BORDER_REPLICATE)
#     return np.reshape(out, img.shape)
#
#
# def rotate_image_as_onehot(img, angle, nlabels, interp=cv2.INTER_LINEAR):
#     onehot_output = rotate_image(convert_to_onehot(img, nlabels=nlabels), angle, interp)
#     return np.argmax(onehot_output, axis=-1)
#
#
# def resize_image(im, size, interp=cv2.INTER_LINEAR):
#     im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
#     # add last dimension again if it was removed by resize
#     if im.ndim > im_resized.ndim:
#         im_resized = np.expand_dims(im_resized, im.ndim)
#     return im_resized
#
#
# def resize_image_as_onehot(im, size, nlabels, interp=cv2.INTER_LINEAR):
#     onehot_output = resize_image(convert_to_onehot(im, nlabels), size, interp=interp)
#     return np.argmax(onehot_output, axis=-1)


def deformation_to_transformation(dx, dy):
    nx, ny = dx.shape

    grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    return map_x, map_y


def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR):
    map_x, map_y = deformation_to_transformation(dx, dy)

    do_optimization = (interp == cv2.INTER_LINEAR)
    # The following command converts the maps to compact fixed point representation
    # this leads to a ~20% increase in speed but could lead to accuracy losses
    # Can be uncommented
    if do_optimization:
        map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)

    remapped = cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT)
    # borderValue=float(np.min(im)))
    if im.ndim > remapped.ndim:
        remapped = np.expand_dims(remapped, im.ndim)
    return remapped


# def dense_image_warp_as_onehot(im, dx, dy, nlabels, interp=cv2.INTER_LINEAR):
#     onehot_output = dense_image_warp(convert_to_onehot(im, nlabels), dx, dy, interp)
#     return np.argmax(onehot_output, axis=-1)
#
#
# def convert_to_onehot(lblmap, nlabels):
#     output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
#     for ii in range(nlabels):
#         output[:, :, ii] = (lblmap == ii).astype(np.uint8)
#     return output
#
#
# def ncc(a, v, zero_norm=True):
#     a = a.flatten()
#     v = v.flatten()
#     if zero_norm:
#         a = (a - np.mean(a)) / (np.std(a) * len(a))
#         v = (v - np.mean(v)) / np.std(v)
#     else:
#         a = a / (np.std(a) * len(a))
#         v = v / np.std(v)
#     return np.correlate(a, v)
#
#
# def norm_l2(a, v):
#     a = a.flatten()
#     v = v.flatten()
#
#     a = (a - np.mean(a)) / (np.std(a) * len(a))
#     v = (v - np.mean(v)) / np.std(v)
#
#     return np.mean(np.sqrt(a**2 + v**2))
#
#
# def all_argmax(arr, axis=None):
#     return np.argwhere(arr == np.amax(arr, axis=axis))
#
#
# def makefolder(folder):
#     """
#     Helper function to make a new folder if doesn't exist
#     :param folder: path to new folder
#     :return: True if folder created, False if folder already exists
#     """
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#         return True
#     return False
#
#
# def load_nii(img_path):
#     """
#     Shortcut to load a nifti file
#     """
#
#     nimg = nib.load(img_path)
#     return nimg.get_data(), nimg.affine, nimg.header
#
#
# def save_nii(img_path, data, affine, header):
#     """
#     Shortcut to save a nifty file
#     """
#
#     nimg = nib.Nifti1Image(data, affine=affine, header=header)
#     nimg.to_filename(img_path)
#
#
# def create_and_save_nii(data, img_path):
#
#     img = nib.Nifti1Image(data, np.eye(4))
#     nib.save(img, img_path)
#
#
# class Bunch:
#     # Useful shortcut for making struct like contructs
#     # Example:
#     # mystruct = Bunch(a=1, b=2)
#     # print(mystruct.a)
#     # >>> 1
#     def __init__(self, **kwds):
#         self.__dict__.update(kwds)
#
#
# def convert_to_uint8(image):
#     image = image - image.min()
#     image = 255.0*np.divide(image.astype(np.float32), image.max())
#     return image.astype(np.uint8)
#
#
# def normalise_image(image):
#     """
#     make image zero mean and unit standard deviation
#     """
#
#     img_o = np.float32(image.copy())
#     m = np.mean(img_o)
#     s = np.std(img_o)
#     return np.divide((img_o - m), s)
#
#
# def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):
#     # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.
#
#     if image.dtype in [np.uint8, np.uint16, np.uint32]:
#         assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o
#
#     if image.dtype == np.uint8:
#         assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o
#
#     min_i = np.percentile(image, 0 + percentiles)
#     max_i = np.percentile(image, 100 - percentiles)
#
#     image = (np.divide((image - min_i), max_i - min_i) * (max_o - min_o) + min_o).copy()
#
#     image[image > max_o] = max_o
#     image[image < min_o] = min_o
#
#     return image
#
#
# def map_images_to_intensity_range(x, min_o, max_o, percentiles=0):
#     x_mapped = np.zeros(x.shape, dtype=np.float32)
#
#     for ii in range(x.shape[0]):
#
#         xc = x[ii, ...]
#         x_mapped[ii, ...] = map_image_to_intensity_range(xc, min_o, max_o, percentiles)
#
#     return x_mapped.astype(np.float32)
#
#
# def normalise_images(x):
#     """
#     Helper for making the images zero mean and unit standard deviation i.e. `white`
#     """
#     x_white = np.zeros(x.shape, dtype=np.float32)
#
#     for ii in range(x.shape[0]):
#
#         xc = x[ii, ...]
#         x_white[ii, ...] = normalise_image(xc)
#
#     return x_white.astype(np.float32)
#
#
# def keep_largest_connected_components(mask):
#     """
#     Keeps only the largest connected components of each label for a segmentation mask.
#     """
#     out_img = np.zeros(mask.shape, dtype=np.uint8)
#
#     for struc_id in [1, 2, 3]:
#
#         binary_img = mask == struc_id
#         blobs = measure.label(binary_img, connectivity=1)
#
#         props = measure.regionprops(blobs)
#
#         if not props:
#             continue
#
#         area = [ele.area for ele in props]
#         largest_blob_ind = np.argmax(area)
#         largest_blob_label = props[largest_blob_ind].label
#
#         out_img[blobs == largest_blob_label] = struc_id
#
#     return out_img
