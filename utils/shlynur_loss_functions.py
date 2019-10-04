import torch
import torch.nn.functional as functional


def cross_entropy2d(network_output, target_output, weight=None, size_average=False):
    """
    A 2D Cross-entropy loss function.
    :param network_output: Predicted segmentation.
    :param target_output: Ground-truth.
    :param weight: A vector containing the weights of the classes.
    :param size_average:
    :return: The loss.
    """
    n, c, h, w = network_output.size()
    log_p = functional.log_softmax(input=network_output, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target_output.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target_output >= 0
    target_output = target_output[mask]
    loss = functional.nll_loss(input=log_p, target=target_output, ignore_index=250, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def cross_entropy3d(network_output, target_output, weight=None, size_average=False):
    """
    A 3D Cross-entropy loss function.
    :param network_output: Predicted segmentation.
    :param target_output: Ground-truth.
    :param weight: A vector containing the weights of the classes.
    :param size_average:
    :return: The loss.
    """
    n, c, d, h, w = network_output.size()
    log_p = functional.log_softmax(input=network_output, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    log_p = log_p[target_output.view(n * d * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target_output >= 0
    target_output = target_output[mask]

    loss = functional.nll_loss(input=log_p, target=target_output, ignore_index=-1, weight=weight, size_average=False)
    # ignore_index=250
    if size_average:
        loss /= mask.data.sum()
    return loss


def soft_dice(pred, target, smoothing=1, non_squared=False):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    if non_squared:
        union = pred.sum() + target.sum()
    else:
        union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
    # if (intersection.item() == 0 and union.item() == 0 and smoothing == 0):
    #     dice = dice.new_tensor([1.0])
    # else:
    dice_score = (2 * intersection + smoothing) / (union + smoothing)

    # fix nans
    dice_score[dice_score != dice_score] = dice_score.new_tensor([1.0])

    return dice_score.mean()


def dice(pred, target):
    pred_bin = (pred > 0.5).float()
    return soft_dice(pred_bin, target, 0, True).item()


def dice_loss(pred, target, non_squared=False):
    return 1 - soft_dice(pred, target, non_squared=non_squared)


def brats_dice_loss(outputs, labels, non_squared=False):
    # bring outputs into correct shape
    # wt, tc, et = outputs.chunk(3, dim=1)
    # s = wt.shape
    # wt = wt.view(s[0], s[2], s[3], s[4])
    # tc = tc.view(s[0], s[2], s[3], s[4])
    # et = et.view(s[0], s[2], s[3], s[4])
    background = outputs[:, 0, :, :, :]
    wt = outputs[:, 1, :, :, :]
    tc = outputs[:, 2, :, :, :]
    et = outputs[:, 3, :, :, :]

    # bring masks into correct shape
    # wt_mask, tc_mask, et_mask = labels.chunk(3, dim=1)
    # s = wt_mask.shape
    # wt_mask = wt_mask.view(s[0], s[2], s[3], s[4])
    # tc_mask = tc_mask.view(s[0], s[2], s[3], s[4])
    # et_mask = et_mask.view(s[0], s[2], s[3], s[4])
    background_mask = (labels == 0).type(torch.cuda.FloatTensor)
    wt_mask = (labels == 1).type(torch.cuda.FloatTensor)
    tc_mask = (labels == 2).type(torch.cuda.FloatTensor)
    et_mask = (labels == 3).type(torch.cuda.FloatTensor)

    # calculate losses
    background_loss = dice_loss(background, background_mask, non_squared=non_squared)
    wt_loss = dice_loss(wt, wt_mask, non_squared=non_squared)
    tc_loss = dice_loss(tc, tc_mask, non_squared=non_squared)
    et_loss = dice_loss(et, et_mask, non_squared=non_squared)
    return (background_loss + wt_loss + tc_loss + et_loss) / 4


def brats_dice_loss_original_4(outputs, labels, non_squared=False):
    output_list = list(outputs.chunk(4, dim=1))
    labels_list = list(labels.chunk(4, dim=0))
    total_loss = 0
    for pred, target in zip(output_list, labels_list):
        total_loss = total_loss + dice_loss(pred, target, non_squared=non_squared)
    return total_loss
