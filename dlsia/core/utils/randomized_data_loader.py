import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def build_randomized_loader(data, labels, batch_size=8, shuffle=True, use_fraction=0.5, pixel_level=True):
    """
    From a single sparsely segmented image, construct two data loaders comprising of a random split of the data.
    Labels to be ignored are marked with -1.

    :param data: input image
    :param labels: input labels. Unused pixels marked with -1
    :param batch_size: batchsize loader
    :param shuffle: shuffle the data?
    :param use_fraction: the fraction of the data to partition
    :param pixel_level: partition data at the pixel level. If false, its split at the tensor level
    :return: a work and test data loader
    """

    if pixel_level:
        draw = torch.rand(labels.shape)
        sel = (draw < use_fraction)
        work_labels = labels.clone().detach()
        test_labels = labels.clone().detach()
        work_labels[~sel] = -1
        test_labels[sel] = -1

        loader_params = {'batch_size': batch_size, 'shuffle': shuffle}
        train_loader = DataLoader(TensorDataset(data, work_labels), **loader_params)
        test_loader = DataLoader(TensorDataset(data, test_labels), **loader_params)

    else:
        draw = torch.rand(labels.shape[0])
        sel = draw < use_fraction
        work_labels = labels[sel]
        test_labels = labels[~sel]
        work_data = data[sel]
        test_data = data[~sel]
        loader_params = {'batch_size': batch_size, 'shuffle': shuffle}
        train_loader = DataLoader(TensorDataset(work_data, work_labels), **loader_params)
        test_loader = DataLoader(TensorDataset(test_data, test_labels), **loader_params)

    return train_loader, test_loader

def build_randomized_loader_regression(data, target, batch_size=8, shuffle=True, use_fraction=0.5):
    """
    From a single sparsely segmented image, construct two data loaders comprising of a random split of the data.
    Labels to be ignored are marked with -1.

    :param target: the target tensor (labels, regression values etc)
    :param data: input image
    :param batch_size: batchsize loader
    :param shuffle: shuffle the data?
    :param use_fraction: the fraction of the data to partition
    :return: a work and test data loader
    """

    N = data.shape[0]
    draw = torch.rand(N)
    sel = (draw < use_fraction)
    work_data = data[sel, ...]
    work_target = target[sel, ...]
    test_data = data[~sel, ...]
    test_target = target[~sel, ...]
    loader_params = {'batch_size': batch_size, 'shuffle': shuffle}
    train_loader = DataLoader(TensorDataset(work_data, work_target), **loader_params)
    test_loader = DataLoader(TensorDataset(test_data, test_target), **loader_params)
    return train_loader, test_loader
