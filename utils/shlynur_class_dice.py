import torch
import numpy as np
from sklearn.metrics import f1_score


def class_dice_3d_prostate(target, prediction, class_num, mode='3d'):
    """
    Calculate the 2D or 3D Dice score of the prostate dataset.
    :param target:      Ground-truth.
    :param prediction:  Predicted labels.
    :param class_num:   Number of classes (unique labels) in the ground-truth.
    :param mode:        Select '2d' to calculate the Dice score over each slice,
                               '3d' to calculate the Dice score over the whole volume.
    :return: The Dice score of the volume.
    """

    if target.is_cuda:
        target = target.cpu()
    if prediction.is_cuda:
        prediction = prediction.cpu()

    if mode == '3d':
        accuracy = np.zeros(class_num)
        target_flattened = torch.flatten(target)
        prediction_flattened = torch.flatten(prediction)
        for c in range(class_num):
            # Look where the label has pixel of the class 'c' and count them
            target_ones = torch.eq(target_flattened, c)
            target_ones_sum = torch.sum(target_ones.float())

            # Look where the prediction has pixel of the class 'c' and count them
            prediction_ones = torch.eq(prediction_flattened, c)
            prediction_ones_sum = torch.sum(prediction_ones.float())

            if (prediction_ones_sum == 0) & (target_ones_sum == 0):  # the class 'c' is in neither
                accuracy[c] = 1
            elif (prediction_ones_sum != 0) & (target_ones_sum == 0):  # the class 'c' is missing in the target
                accuracy[c] = 0
            # elif (prediction_ones_sum == 0) & (target_ones_sum != 0):
            #     accuracy[c] = 0
            else:
                acc = f1_score(target_flattened, prediction_flattened, labels=[c], average=None)
                accuracy[c] = acc[0]
        return accuracy
    elif mode == '2d':
        accuracy = np.zeros((target.shape[0], class_num))
        for slice_no in range(target.shape[0]):
            target_s = target[slice_no, :, :]
            target_s = torch.flatten(target_s)
            prediction_s = prediction[slice_no, :, :]
            prediction_s = torch.flatten(prediction_s)

            for c in range(class_num):
                # Look where the label has pixel of the class 'c' and count them
                target_ones = torch.eq(target_s, c)
                target_ones_sum = torch.sum(target_ones.float())

                # Look where the prediction has pixel of the class 'c' and count them
                prediction_ones = torch.eq(prediction_s, c)
                prediction_ones_sum = torch.sum(prediction_ones.float())

                if (prediction_ones_sum == 0) & (target_ones_sum == 0):  # the class 'c' is in neither
                    accuracy[slice_no, c] = 1
                elif (prediction_ones_sum != 0) & (target_ones_sum == 0):  # the class 'c' is missing in the target
                    accuracy[slice_no, c] = 0
                # elif (prediction_ones_sum == 0) & (target_ones_sum != 0):
                #     accuracy[slice_no, c] = 0
                else:
                    acc = f1_score(target_s, prediction_s, labels=[c], average=None)
                    accuracy[slice_no, c] = acc[0]
        return np.array(np.mean(accuracy, axis=0))
    else:
        raise ValueError('The selected mode must be either "2D" or "3D".')


def class_dice_3d_brats(target, prediction):
    """
    Calculate the 3D Dice score of the BraTS dataset.
    :param target:      Ground-truth.
    :param prediction:  Predicted labels.
    :return:            The Dice scores of the three nested tumor regions.
    """
    if target.is_cuda:
        target = target.cpu()
    if prediction.is_cuda:
        prediction = prediction.cpu()

    target_flattened = torch.flatten(target)
    prediction_flattened = torch.flatten(prediction)

    # Calculate the whole tumor Dice score.
    target_wt = target_flattened > 0
    prediction_wt = prediction_flattened > 0
    wt = f1_score(target_wt, prediction_wt, average='binary')

    # Calculate the tumor core Dice score.
    target_tc = torch.zeros_like(target_flattened)
    prediction_tc = torch.zeros_like(prediction_flattened)
    for i in (1, 3, 4):
        target_tc = torch.where(target_flattened == i, target_flattened, target_tc)
        prediction_tc = torch.where(prediction_flattened == i, prediction_flattened, prediction_tc)
    target_tc = target_tc > 0
    prediction_tc = prediction_tc > 0
    tc = f1_score(target_tc, prediction_tc, average='binary')

    # Calculate the enhancing tumor Dice score.
    target_et = target_flattened == 4
    prediction_et = prediction_flattened == 4
    et = f1_score(target_et, prediction_et, average='binary')

    total_score = [wt, tc, et]

    return total_score
