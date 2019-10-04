import os
import torch
import torch.nn as nn
import numpy as np
# from torch.autograd import Variable
from utils.shlynur_class_dice import class_dice_3d_prostate, class_dice_3d_brats
# from utils.shlynur_loss_functions import cross_entropy3d
from skimage import measure


def print_memory_usage(status=''):
    """
    Prints the current GPU memory usage.
    :param status:  Text to explain the current situation.
    :return:        Current allocated GPU memory.
    """
    print("{}: \t{:.4f} GB".format(status, (torch.cuda.memory_allocated() * (10 ** -9))))


def convergence_check_auto_cnn(cnn, bool_prostate_data, number_of_classes, data_loader, iter_number=4):
    """
    Calculate the validation accuracy and loss of the AutoCNN model in-between epochs while training the AutoCNN.
    :param cnn:                 The AutoCNN model.
    :param bool_prostate_data:  True if using the prostate data, False if using the BraTS data.
    :param number_of_classes:   Number of unique labels in the segmentation.
    :param data_loader:         The PyTorch DataLoader class.
    :param iter_number:         Number of sub-volumes to create from each BraTS volume.
    :return:                    Validation Dice score, validation loss.
    """
    cnn.eval()  # Change the model to 'eval' mode.

    val_loss = 0

    # Loss and activation functions.
    criterion = nn.CrossEntropyLoss()
    activation = nn.Softmax(dim=1)

    average_classes, mean_average_classes = [], []
    for val_images, val_labels in data_loader:
        if bool_prostate_data:
            # Using the prostate dataset.

            # Add a dimension to the prostate image (1 channel)
            val_images = val_images.unsqueeze(1)

            val_images = val_images.float().cuda()
            val_labels = val_labels.long().cuda()

            # Get the AutoCNN output.
            val_outputs = cnn(val_images)
            del val_images

            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()

            # Softmax of the AutoCNN output.
            max_in = activation(val_outputs)
            max_in = torch.max(max_in, 1)[1].long()

            # Calculate the Dice score for each class
            classes_score = class_dice_3d_prostate(target=val_labels.data[0, :, :, :],
                                                   prediction=max_in[0, :, :, :], class_num=number_of_classes)
            average_classes.append(classes_score)

            del val_labels, val_outputs, max_in
        else:
            # Using the BraTS dataset.
            slice_spacing = 160 // iter_number
            for i in range(iter_number):
                # Create 40-slice sub-volumes.
                new_val_images = val_images[:, :, i * slice_spacing:(i + 1) * slice_spacing, :].float().cuda()
                new_labels = val_labels[:, i * slice_spacing:(i + 1) * slice_spacing, :].long().cuda()

                # Get the AutoCNN output.
                new_val_outputs = cnn(new_val_images)
                del new_val_images

                loss = criterion(new_val_outputs, new_labels)
                val_loss += loss.item()

                # Softmax of the AutoCNN output.
                max_in = activation(new_val_outputs)
                max_in = torch.max(max_in, 1)[1].long()

                # Calculate the Dice score for the three tumor regions.
                classes_score = class_dice_3d_brats(target=new_labels.data[0, :, :, :],
                                                    prediction=max_in[0, :, :, :])
                average_classes.append(classes_score)
                del new_val_outputs, max_in

        del val_images

    total_loss = val_loss / len(data_loader.dataset)
    if not bool_prostate_data:
        total_loss = total_loss / iter_number

    average_classes = np.squeeze(np.array(average_classes))
    if bool_prostate_data:
        mean_average_classes = np.mean(average_classes, axis=0)
    else:
        mean_average_classes = np.mean(np.mean(average_classes, axis=0), axis=0)

    if len(data_loader.dataset) == 1:
        return average_classes, val_loss
    else:
        return mean_average_classes, total_loss


def prediction_converter(outputs):
    """
    Convert the CNN outputs to predicted labels via a softmax activation function.
    :param outputs: Output of the CNN model.
    :return:        predicted segmentation mask.
    """
    activation = nn.Softmax(dim=1)
    max_in = activation(outputs)
    max_in = torch.max(max_in, 1)[1]
    prediction = max_in.data.cpu()
    return prediction


def prediction_converter_without_max(outputs):
    """
    Apply a softmax activation function to the CNN outputs.
    :param outputs: Outputs of the CNN model.
    :return:        Softmax probability distribution of the CNN outputs.
    """
    activation = nn.Softmax(dim=1)
    max_in = activation(outputs)
    prediction = max_in.data.cpu()
    return prediction


def identify_incorrect_pixels(target, prediction, class_num):
    """
    Convert the ground-truth and predictions to tensors with an additional channel of size 'class_num' in each tensor.
    In every channel:
        The new ground-truth is 1 when the original mask has the corresponding class label, 0 elsewhere.
        The new prediction is 0 when the original prediction is correct when compared to the ground-truth, 0
            elsewhere.
    :param target:      Ground-truth.
    :param prediction:  Predicted labels.
    :param class_num:   Number of classes (unique labels) in the data.
    :return:            the converted incorrect prediction and labels tensors.
    """
    # Set the correctly labeled data to 0 and the incorrectly labeled data to 1
    incorrect_pred = torch.ne(target, prediction).type(torch.LongTensor)

    new_labels = torch.zeros(size=((class_num,) + target.shape)).type(torch.LongTensor)
    new_labels.transpose_(0, 1)

    for i in range(class_num):
        new_labels[:, i, :] = torch.eq(target, i)
    incorrect_predictions = torch.mul(new_labels, incorrect_pred)

    return incorrect_predictions, new_labels


def make_gaussian_2d(size, sigma=10, center=None):
    """
    Function that makes 2D Gaussian-shaped scribbles around the selected incorrectly classified pixel.
    :param size:    Size of the image.
    :param sigma:   Width of the Gaussian function.
    :param center:  Location of the pixel.
    :return:        A 2D Gaussian curve around a selected pixel.
    """
    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)
    x = x[:, np.newaxis]

    if center is None:
        x0 = size[0] // 2
        y0 = size[1] // 2
    else:
        [x0, y0] = center

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(np.float64)


def make_gaussian_3d(size, sigma=10, center=None):
    """
    Function that makes 3D Gaussian-shaped scribbles around the selected incorrectly classified pixel.
    :param size:    Size of the image.
    :param sigma:   Width of the Gaussian function.
    :param center:  Location of the pixel.
    :return:        A 3D Gaussian curve around a selected pixel.
    """
    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)
    z = np.arange(0, size[2], 1, float)
    x = x[:, np.newaxis, np.newaxis]
    y = y[:, np.newaxis]

    if center is None:
        x0 = size[0] // 2
        y0 = size[1] // 2
        z0 = size[2] // 2
    else:
        [x0, y0, z0] = center

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / sigma ** 2).astype(np.float64)


def mask_creator(incorrect, label, class_num, mode='2d', number_of_slices='all', number_of_scribbles=1,
                 size_of_scribble=4, bool_best_placement=False):
    """
    Create the scribble mask that will be fed into the InterCNN.
    :param incorrect:           Incorrect prediction (created by the function 'identify_incorrect_pixels').
    :param label:               Ground-truth (created by the function 'identify_incorrect_pixels').
    :param class_num:           Number of classes (unique labels) in the ground-truth.
    :param mode:                Either "2d" or "3d" for a 2D or 3D gaussian scribble.
    :param number_of_slices:    Place scribbles in 'one' slice or 'all' slices.
    :param number_of_scribbles: Number of scribbles to be created per class in one iteration.
    :param size_of_scribble:    Size of the gaussian scribble (value of sigma).
    :param bool_best_placement: True to place the scribbles in the largest connected component, False to ignore.
    :return:
    """

    scribbles_array = np.zeros(label.shape)
    gaussian_size_2d = label.shape[-2:]
    gaussian_size_3d = label.shape[-3:]

    if mode == '2d':
        # Create 2D Gaussian scribbles.
        if number_of_slices == 'one':
            # Place 2D Gaussian scribbles in the middle slice (p).
            p = label.shape[2] // 2
            for s in range(class_num):
                incorrect_p = incorrect[:, s, p, :, :].squeeze_()
                incorrect_p = incorrect_p.numpy()
                incorrect_pixel_locations = np.where(incorrect_p == 1)

                length = len(incorrect_pixel_locations[0])

                if length > 0:
                    random_pixel = np.random.randint(length)
                    pixel_coord = (incorrect_pixel_locations[0][random_pixel],
                                   incorrect_pixel_locations[1][random_pixel])
                    scribbles = make_gaussian_2d(size=gaussian_size_2d, sigma=size_of_scribble, center=pixel_coord)
                    scribbles_array[0, s, p, :, :] = scribbles
        elif number_of_slices == 'all':
            # Place 2D Gaussian scribbles in every slice of the volume.
            for s in range(class_num):
                for p in range(label.shape[2]):
                    incorrect_p = incorrect[:, s, p, :, :].squeeze_()
                    incorrect_p = incorrect_p.numpy()
                    incorrect_pixel_locations = np.where(incorrect_p == 1)

                    length = len(incorrect_pixel_locations[0])

                    if length > 0:
                        random_pixel = np.random.randint(length)
                        pixel_coord = (incorrect_pixel_locations[0][random_pixel],
                                       incorrect_pixel_locations[1][random_pixel])
                        scribbles = make_gaussian_2d(size=gaussian_size_2d, sigma=size_of_scribble, center=pixel_coord)
                        scribbles_array[0, s, p, :, :] = scribbles
        else:
            raise ValueError('The number_of_slices variable must be either "all" or "one"')
    elif mode == '3d':
        # Create 3D Gaussian scribbles.
        for s in range(class_num):
            incorrect_p = incorrect[0, s, :]
            incorrect_p = incorrect_p.numpy()
            incorrect_pixel_locations = np.where(incorrect_p == 1)
            number_of_pixels = len(incorrect_pixel_locations[0])

            if number_of_pixels >= number_of_scribbles:
                # Place 1 or more scribbles in the volume.

                if bool_best_placement:
                    # Find the biggest connected component
                    connected_labels = measure.label(incorrect_p, connectivity=1, return_num=False)
                    unique, counts = np.unique(connected_labels, return_counts=True)
                    unique = unique[1:]
                    counts = counts[1:]
                    locations = np.where(connected_labels == unique[np.argmax(counts)])
                    length = len(locations[0])
                else:
                    locations = incorrect_pixel_locations
                    length = number_of_pixels

                total_random_pixels = np.random.choice(length, size=number_of_scribbles, replace=False)
                for i in range(number_of_scribbles):
                    random_pixel = total_random_pixels[i]
                    pixel_coord = (locations[0][random_pixel],
                                   locations[1][random_pixel],
                                   locations[2][random_pixel])
                    print(pixel_coord)
                    scribbles = make_gaussian_3d(size=gaussian_size_3d, sigma=size_of_scribble, center=pixel_coord)
                    scribbles_array[0, s, :, :, :] += scribbles
    else:
        raise ValueError('The selected mode must be either "2d" or "3d"')

    return scribbles_array


def accuracy_check_auto_inter_cnn(auto_cnn, inter_cnn, class_num, number_of_slices, max_iterations, bool_prostate_data,
                                  data_loader, save_folder, model_type, num_epochs_inter_cnn, num_filters_inter_cnn,
                                  learning_rate, bool_data_augmentation, bool_loss_summation=False,
                                  accumulation_steps=5, mode='2d', number_of_scribbles=1, size_of_scribble=4,
                                  bool_best_placement=False, iter_number=4):
    """
    Calculates the AutoCNN and InterCNN validation scores in-between epochs while training the InterCNN.
    :param auto_cnn:                AutoCNN model.
    :param inter_cnn:               InterCNN model.
    :param class_num:               Number of classes (unique labels) in the ground-truth.
    :param number_of_slices:        Place scribbles in 'one' slice or 'all' slices.
    :param max_iterations:          Number of validation iterations.
    :param bool_prostate_data:      True when using the prostate data, False when using the BraTS data.
    :param data_loader:             The PyTorch DataLoader class.
    :param save_folder:             The directory to save
    :param model_type:              Either 'gustav' or 'robin'
    :param num_epochs_inter_cnn:    Number of the InterCNN epochs
    :param num_filters_inter_cnn:   Number of filters in the InterCNN model.
    :param learning_rate:           The learning rate of the InterCNN optimizer.
    :param bool_data_augmentation:  Enable / disable data augmentation.
    :param bool_loss_summation:     Enable / disable the loss summation
    :param accumulation_steps:      Number of times that the loss is summed, i.e. that the model graph is saved to
                                        accumulate the gradients and backpropagate afterwards.
    :param mode:                    Either "2d" or "3d" for a 2D or 3D gaussian scribble.
    :param number_of_scribbles:     Number of scribbles to be created per class in one iteration.
    :param size_of_scribble:        Size of the gaussian scribble (value of sigma).
    :param bool_best_placement:     True to place the scribbles in the largest connected component, False to ignore.
    :param iter_number:             Number of sub-volumes to create from each BraTS volume.
    :return: AutoCNN validation score, InterCNN validation score, InterCNN validation loss and validation score of the
             last input image over the iterations.
    """
    # Enable evaluation mode on the models.
    auto_cnn.eval()
    inter_cnn.eval()

    # Loss and activation functions.
    criterion = nn.CrossEntropyLoss()
    val_activation = nn.Softmax(dim=1)

    average_class_score_auto_cnn, average_class_score_inter_cnn = [], []
    running_loss = 0
    counter = 0
    for i, (val_images, val_labels) in enumerate(data_loader):
        class_scores_inter_cnn_iterations = []
        if bool_prostate_data:
            # Using the prostate dataset.

            # Add a dimension to the prostate image (1 channel)
            val_images = val_images.unsqueeze(1)

            val_labels_copy = val_labels.long()
            val_images = val_images.float().cuda()
            val_labels = val_labels.long().cuda()

            # Get the AutoCNN prediction
            val_outputs_auto_cnn = auto_cnn(val_images)
            prediction_auto_cnn = prediction_converter(outputs=val_outputs_auto_cnn)
            if bool_loss_summation:
                prediction_torch = val_activation(val_outputs_auto_cnn)  # .detach().clone())
            else:
                prediction_torch = prediction_auto_cnn.unsqueeze(1)
            del val_outputs_auto_cnn

            # Calculate class scores for autoCNN
            classes_score_auto = class_dice_3d_prostate(
                target=val_labels_copy[0, :, :, :], prediction=prediction_auto_cnn[0, :, :, :], class_num=class_num)
            average_class_score_auto_cnn.append(classes_score_auto)

            # Create scribbles for input to interCNN
            incorrect_predictions, new_val_labels = identify_incorrect_pixels(
                target=val_labels_copy, prediction=prediction_auto_cnn, class_num=class_num)
            scribbles = mask_creator(
                incorrect=incorrect_predictions, label=new_val_labels, class_num=class_num,
                number_of_slices=number_of_slices, mode=mode, number_of_scribbles=number_of_scribbles,
                size_of_scribble=size_of_scribble, bool_best_placement=bool_best_placement)

            prediction_torch = prediction_torch.float().cuda()
            scribbles_torch = torch.from_numpy(scribbles)
            scribbles_torch = scribbles_torch.float().cuda()

            loss_adder = torch.zeros(1)

            for index in range(max_iterations):
                # Make a new prediction
                network_inputs = torch.cat((val_images, prediction_torch, scribbles_torch), dim=1)
                val_outputs_inter_cnn = inter_cnn(network_inputs)

                del network_inputs, prediction_torch, scribbles_torch

                prediction_inter_cnn = prediction_converter(outputs=val_outputs_inter_cnn)
                if bool_loss_summation:
                    prediction_torch = val_activation(val_outputs_inter_cnn)  # .detach().clone())
                else:
                    prediction_torch = prediction_inter_cnn.unsqueeze(1)

                loss = criterion(val_outputs_inter_cnn, val_labels)

                del val_outputs_inter_cnn

                if bool_loss_summation:
                    loss_adder += loss

                    # Backpropagate and calculate the loss.
                    if (index + 1) % accumulation_steps == 0:
                        loss_adder.backward()
                        running_loss += loss_adder.item()
                        counter += 1
                else:
                    loss.backward()
                    running_loss += loss.item()
                    counter += 1

                # Calculate the class scores for InterCNN
                classes_score_inter_new = class_dice_3d_prostate(target=val_labels_copy[0, :, :, :],
                                                                 prediction=prediction_inter_cnn[0, :, :, :],
                                                                 class_num=class_num)
                class_scores_inter_cnn_iterations.append(classes_score_inter_new)

                # Create new scribbles
                incorrect_predictions, new_val_labels = identify_incorrect_pixels(
                    target=val_labels_copy, prediction=prediction_inter_cnn, class_num=class_num)
                scribbles = scribbles + mask_creator(
                    incorrect=incorrect_predictions, label=new_val_labels, class_num=class_num,
                    number_of_slices=number_of_slices, mode=mode, number_of_scribbles=number_of_scribbles,
                    size_of_scribble=size_of_scribble, bool_best_placement=bool_best_placement)

                prediction_torch = prediction_torch.float().cuda()
                scribbles_torch = torch.from_numpy(scribbles)
                scribbles_torch = scribbles_torch.float().cuda()

            average_class_score_inter_cnn.append(class_scores_inter_cnn_iterations)
            del val_images, val_labels, prediction_torch, scribbles_torch
        else:
            # Using the BraTS dataset.
            slice_spacing = 160 // iter_number
            for a in range(iter_number):
                # Create 40-slice sub-volumes.
                new_val_images = val_images[:, :, a * slice_spacing:(a + 1) * slice_spacing, :].float().cuda()
                new_val_labels = val_labels[:, a * slice_spacing:(a + 1) * slice_spacing, :].long().cuda()
                new_val_labels_copy = val_labels[:, a * slice_spacing:(a + 1) * slice_spacing, :].long()

                # Get the AutoCNN prediction
                new_val_outputs_auto_cnn = auto_cnn(new_val_images)
                prediction_auto_cnn = prediction_converter(outputs=new_val_outputs_auto_cnn)
                if bool_loss_summation:
                    prediction_torch = val_activation(new_val_outputs_auto_cnn)  # .detach().clone())
                else:
                    prediction_torch = prediction_auto_cnn.unsqueeze(1)
                del new_val_outputs_auto_cnn

                # Calculate Dice scores for AutoCNN
                classes_score_auto = class_dice_3d_brats(target=new_val_labels_copy[0, :, :, :],
                                                         prediction=prediction_auto_cnn[0, :, :, :])
                average_class_score_auto_cnn.append(classes_score_auto)

                # Create scribbles for input to interCNN
                incorrect_predictions, new_new_val_labels = identify_incorrect_pixels(
                    target=new_val_labels_copy, prediction=prediction_auto_cnn, class_num=class_num)
                scribbles = mask_creator(
                    incorrect=incorrect_predictions, label=new_new_val_labels, class_num=class_num,
                    number_of_slices=number_of_slices, mode=mode, number_of_scribbles=number_of_scribbles,
                    size_of_scribble=size_of_scribble, bool_best_placement=bool_best_placement)

                prediction_torch = prediction_torch.float().cuda()
                scribbles_torch = torch.from_numpy(scribbles)
                scribbles_torch = scribbles_torch.float().cuda()

                loss_adder = torch.zeros(1)

                for index in range(max_iterations):
                    # Make a new prediction
                    network_inputs = torch.cat((new_val_images, prediction_torch, scribbles_torch), dim=1)
                    val_outputs_inter_cnn = inter_cnn(network_inputs)

                    del network_inputs, prediction_torch, scribbles_torch

                    prediction_inter_cnn = prediction_converter(outputs=val_outputs_inter_cnn)
                    if bool_loss_summation:
                        prediction_torch = val_activation(val_outputs_inter_cnn)  # .detach().clone())
                    else:
                        prediction_torch = prediction_inter_cnn.unsqueeze(1)

                    loss = criterion(val_outputs_inter_cnn, new_val_labels)

                    del val_outputs_inter_cnn

                    if bool_loss_summation:
                        loss_adder += loss

                        # Backpropagate and calculate the loss.
                        if (index + 1) % accumulation_steps == 0:
                            loss_adder.backward()
                            running_loss += loss_adder.item()
                            counter += 1
                    else:
                        loss.backward()
                        running_loss += loss.item()
                        counter += 1

                    # Calculate the class scores for interCNN
                    classes_score_inter_new = class_dice_3d_prostate(target=new_val_labels_copy[0, :, :, :],
                                                                     prediction=prediction_inter_cnn[0, :, :, :],
                                                                     class_num=class_num)
                    class_scores_inter_cnn_iterations.append(classes_score_inter_new)

                    # Create new scribbles
                    incorrect_predictions, new_new_val_labels = identify_incorrect_pixels(
                        target=new_val_labels_copy, prediction=prediction_inter_cnn, class_num=class_num)
                    scribbles = scribbles + mask_creator(
                        incorrect=incorrect_predictions, label=new_new_val_labels, class_num=class_num,
                        number_of_slices=number_of_slices, mode=mode, number_of_scribbles=number_of_scribbles,
                        size_of_scribble=size_of_scribble, bool_best_placement=bool_best_placement)

                    prediction_torch = prediction_torch.float().cuda()
                    scribbles_torch = torch.from_numpy(scribbles)
                    scribbles_torch = scribbles_torch.float().cuda()

                average_class_score_inter_cnn.append(class_scores_inter_cnn_iterations)
                del new_val_images, new_val_labels, prediction_torch, scribbles_torch

        np.save(
            os.path.join(save_folder, 'iterations',
                         'intercnn_3D_model_{}_data_{}_epochs_{}_filters_{}_lr_{}_dataug_{}_val_scores_sample_{}'
                         .format(model_type, 'prostate' if bool_prostate_data else 'brats', num_epochs_inter_cnn,
                                 num_filters_inter_cnn, learning_rate, bool_data_augmentation, i)),
            class_scores_inter_cnn_iterations)

        del val_images, val_labels, prediction_torch, scribbles_torch

    total_loss = running_loss / counter

    average_class_score_auto_cnn = np.squeeze(np.array(average_class_score_auto_cnn))
    average_class_score_inter_cnn = np.squeeze(np.array(average_class_score_inter_cnn))

    if len(data_loader.dataset) == 1:
        final_auto_cnn_score = average_class_score_auto_cnn
        if max_iterations == 1:
            final_inter_cnn_score = average_class_score_inter_cnn
        else:
            final_inter_cnn_score = np.mean(average_class_score_inter_cnn, axis=0)
    else:
        final_auto_cnn_score = np.mean(average_class_score_auto_cnn, axis=0)
        if max_iterations == 1:
            final_inter_cnn_score = np.mean(average_class_score_inter_cnn, axis=0)
        else:
            final_inter_cnn_score = np.mean(np.mean(average_class_score_inter_cnn, axis=0), axis=0)

    return final_auto_cnn_score, final_inter_cnn_score, total_loss, class_scores_inter_cnn_iterations
