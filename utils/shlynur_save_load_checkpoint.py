import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, val_score, model_name):
    """
    Save the model and optimizer parameters.
    :param model: model.
    :param optimizer: optimizer.
    :param scheduler: learning rate scheduler.
    :param epoch: current epoch.
    :param val_score: current validation score.
    :param model_name: name of the model.
    """
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'validation score': val_score
    }, model_name)


def load_checkpoint(model, optimizer=None, scheduler=None, file_folder='', filename=''):
    """
    Load the model and optimizer parameters.
    :param model: model.
    :param optimizer: optimizer.
    :param scheduler: learning rate scheduler.
    :param file_folder: save folder.
    :param filename: name of the model.
    """
    print("=> Loading model {}".format(filename))
    checkpoint = torch.load(os.path.join(file_folder, filename))
    starting_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is None:
        optimizer = []
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is None:
        scheduler = []
    else:
        scheduler.load_state_dict(checkpoint['scheduler'])
    val_score = checkpoint['validation score']

    return model, optimizer, scheduler, starting_epoch, val_score
