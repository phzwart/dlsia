import numpy as np
import torch
import torch.nn.utils


# TODO: Fill out to support tutorials (EJR 12/01/22)

def inference_segmentation(net, testloader, device):
    '''
    Description coming
    '''

    for data in testloader:
        image, target = data  # load noisy and clean images from loader

        image = image.to(device)
        target = target.to(device)

        # forward pass
        output = net(image)

        output = torch.max(output, 1)

    return output


def inference_classification(net, testloader, device):
    """
    ERIC, PLEASE ADD DESCRIPTION

    :param net:
    :type net:
    :param testloader:
    :type testloader:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    correct = 0
    wrong = 0
    total = 0

    for data in testloader:
        image, target = data  # load noisy and clean images from loader

        image = image.to(device)
        target = target.to(device)

        # forward pass
        output = net(image)

        output = torch.max(output, 1)

        correct += torch.sum(target == output.indices)
        wrong += (len(target) - torch.sum(target == output.indices))
        total += len(target)
    return correct, wrong, total
