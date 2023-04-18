import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch
import torchvision.transforms.functional as TF
from torch.autograd import Variable
from torch.nn import Module, Conv2d, Conv3d
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader


"""
This modules contains various helper functions assisting in:
    1) making directories and retrieving devices,
    2) counting network parmeters
    3) saving images
    4) measuring training performance
"""


def get_device():
    """
    :return: either an available GPU or the CPU
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def make_dir(dir_name):
    """
    Make a directory for storing reconstructed images

    :param str dir_name: user specified name of new directory
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_loader(x, y,
                 batch_size,
                 shuffle=True,
                 pin_memory=True,
                 drop_last=True):
    """
    Loads image data into PyTorch DataLoader class, allowing for easy handling
    and iterative loading of data data into the networks and models.

    :param x: numpy array of input image data
    :param y: numpy array of target image data
    :param batch_size: number of images loaded into a single batch for
                       network processing
    :param bool shuffle: if True, data is randomly shuffled at each epoch
    :param bool pin_memory: if True, host=to-device (CPU-to-GPU) data
                            transfer is faster
    :param bool drop_last: if True, final incomplete batch is dropped if
                           number of images in dataset is not divisible by
                           batch_size
    """

    loader_params = {'batch_size': batch_size,
                     'shuffle': shuffle,
                     'num_workers': 0,
                     'pin_memory': pin_memory,
                     'drop_last': drop_last}

    # Loaders are created below
    data = TensorDataset(torch.Tensor(x), torch.Tensor(y))

    loader = DataLoader(data, **loader_params)

    return loader


def count_parameters(module: Module):
    """
    Counts the number of learnable parameters in the network

    :param module: Module: the defined network, usually passed to the
                    variable 'net'
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_conv2d(module: Module):
    """
    Counts the number of layers in the network

    :param module: Module: the defined network, usually passed to the
                    variable 'net'
    """
    return len([m for m in module.modules() if isinstance(m, Conv2d)])


def count_conv3d(module: Module):
    """
    Counts the number of layers in the network

    :param module: Module: the defined network, usually passed to the
                    variable 'net'
    """
    return len([m for m in module.modules() if isinstance(m, Conv3d)])


def get_in_channels(module: Module):
    """
    Counts the number of incoming channels in the initial 2d or 3d
    convolutional layer

    :param module: Module: the defined network, usually passed to the
                    variable 'net'
    """
    in_channels = 0
    for m in module.modules():
        if isinstance(m, Conv2d):
            in_channels = m.in_channels
            break

    # If no 2d convolutions, check for 3d
    if in_channels == 0:
        for m in module.modules():
            if isinstance(m, Conv3d):
                in_channels = m.in_channels
                break

    return in_channels


def get_out_channels(module: Module):
    """
    Counts the number of outgoing channels in the final 2d or 3d
    convolutional layer

    :param module: Module: the defined network, usually passed to the
                    variable 'net'
    """
    out_channels = 0
    for m in module.modules():
        if isinstance(m, Conv2d):
            out_channels = m.out_channels

    # If no 2d convolutions, check for 3d
    if out_channels == 0:
        for m in module.modules():
            if isinstance(m, Conv3d):
                out_channels = m.out_channels

    return out_channels


def binary_acc(y_pred, y_target):
    """
    Calculates accuracy of prediction by rounding each output pixel to
    zero or 1.Inputs are same as criterion evaluation function

    :param y_pred: prediction resulting running images through model
    :param y_target: target image in which the prediction is scored
                     against
    """

    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_target).sum().float()
    acc = correct_results_sum / np.prod(list(y_target.size()))

    return acc


def save_decoded_image2d(img, name):
    """
    Saves torch images (clean, noisy, or reconstructed)

    :param img: The torch image to be saved, typically extracted from a
                batch of test data
    :param name: Name of image (with existing directory path) in which
                 images are saved
    """

    img = img.view(img.size(0), 1, img.size(-2), img.size(-1))
    save_image(img, name)


def save_training_images(noisy, output, mask, path, epoch, train=True):
    """
    Plots training/validation tensor images in hopes of tracking
    progress within the training loop
    
    :param noisy: noisy image tensor
    :type noisy: List[float]
    :param output: model output tensor
    :type output: List[float]
    :param mask: binary mask tensor using 0 for background and 1 for
                 peak class
    :type mask: List[float]
    :param str path: directory in which images are saved to
    :param int epoch: epoch number appended to image name
    :param bool train: True if training images, False if validation images
    """

    output_round = torch.round(output)

    plt.figure(figsize=(22, 5))
    plt.subplot(141)
    plt.imshow(noisy.detach().numpy())
    plt.colorbar(shrink=0.8)
    plt.title('Noisy')
    plt.subplot(142)
    plt.imshow(output.detach().numpy())
    plt.colorbar(shrink=0.8)
    plt.title('Prediction')
    plt.subplot(143)
    plt.imshow(output_round.detach().numpy())
    plt.colorbar(shrink=0.8)
    plt.title('Rounded')
    plt.subplot(144)
    plt.imshow(mask.detach().numpy())
    plt.colorbar(shrink=0.8)
    plt.title('Mask')

    plt.rcParams.update({'font.size': 20})
    plt.tight_layout()

    if train:

        save_name = './' + path + '/train_epoch{}.png'.format(epoch + 1)
        plt.savefig(save_name)
        plt.show()
    else:
        save_name = './' + path + '/val_epoch{}.png'.format(epoch + 1)
        plt.savefig(save_name)
        plt.show()


def save_loss_images(train_loss, validation_loss, train_acc, validation_acc,
                     path, log_plot=True):
    """
    Plots loss and binary accuracy functions for both the training and
    validation sets. Accuracy arrays computed via helpers.binary_acc

    :param log_plot: Toggle whether or not we want the logarithm of the loss
    :type log_plot: Bool
    :param train_loss: training loss as a function of epoch
    :param validation_loss: validation set loss as a function of epoch
    :param train_acc: training accuracy as a function of epoch
    :param validation_acc: validation set accuraccy as a function of
                           epoch
    :param path: path in which to save image
    """
    plt.figure(figsize=(18, 5))
    plt.subplot(121)
    plt.plot(train_loss, linewidth=3, label='training loss')
    plt.plot(validation_loss, linewidth=3, label='validation loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if log_plot is True:
        plt.yscale('log')
    plt.legend()

    plt.subplot(122)
    plt.plot(train_acc, linewidth=3, label='training accuracy')
    plt.plot(validation_acc, linewidth=3, label='validation accuracy')
    plt.hlines(1, 0, len(train_loss), colors='k', linestyles='dashed')
    plt.title('Binary Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Ratio Correct')

    plt.rcParams.update({'font.size': 16})
    plt.legend()

    axes = plt.gca()
    axes.set_ylim([.95, 1])

    plt.tight_layout()

    plt.savefig('./' + path + '/loss_accuracy.png')


def save_image_reconstructions2d(net, testloader, dir_name):
    """
    A simple module for running noisy images from the testing set
    through the model. A batch of original, noisy, and cleaned versions
    of all pics are saved.

    :param net: the trained network
    :param testloader: set of data to test, created via torch.utils.data
                       import DataLoader
    :param dir_name: directory in which all images are saved
    """

    device = get_device()
    net.to(device)

    counter = 0  # for saved image indexing
    for batch in testloader:
        noisy, clean = batch

        noisy = noisy.type(torch.FloatTensor)
        clean = clean.type(torch.FloatTensor)
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = net(noisy)

        # print tensor max/min for debugging
        # print('min/max clean:', torch.min(clean), torch.max(clean))
        # print('min/max noisy:', torch.min(noisy), torch.max(noisy))
        # print('min/max output:', torch.min(output), torch.max(output))

        # Save clean, noisy, and output images below
        noisy_name = './' + dir_name \
                     + '/peaks_noisy{}.png'.format(counter)
        clean_name = './' + dir_name \
                     + '/peaks_original{}.png'.format(counter)
        output_name = './' + dir_name \
                      + '/peaks_reconstructed{}.png'.format(counter)

        save_decoded_image2d(noisy.cpu().data, name=noisy_name)
        save_decoded_image2d(output.cpu().data, name=output_name)
        save_decoded_image2d(clean.cpu().data, name=clean_name)

        counter = counter + 1


def create_class_mapping(info, get_label):
    """
    Create a mapping from class names to indices
    Automatically creates a class called background that has index 0

    :param info:    A sequence where the entry i contains descriptions of the 
                    ith image                
    :type info:     array-like
    :param get_label: A one-parameter function that takes in an entry of info, 
                    say info[i], and returns the associated label (the class of 
                    the ith image)
    :type get_label: function(Object)
    """
    labels = np.array(list(map(get_label, info)))
    unique_labels = np.unique(labels)
    mapping = {unique_labels[i]: i + 1 for i in range(len(unique_labels))}
    mapping["background"] = 0  # background will have index 0
    return mapping


def create_mask(tens, top, left, height, width, label):
    """
    Create a mask cropped from the given tensor with the specified label 

    :param torch.FloatTensor tens: tensor to extract the mask from
    :param int top: vertical component of the top left corner of the crop
    :param int left: horizontal coordinate of the top left corner of the crop
    :param int height: height of the crop
    :param int width: width of the crop
    :param int label: label to which all nonzero pixel values are converted
    """

    tens = TF.crop(tens, top, left, height, width)  # crop to the desired size
    tens = torch.sum(tens, dim=0, keepdim=True)  # scale down to 1 channel

    # convert all nonzero values to have the value label
    tens = torch.where(tens != 0., float(label), 0.)
    return tens


class RandomCrop_Dataset(torch.utils.data.Dataset):
    """
    Create a dataset of samples from a set of images and masks by first
    filtering the set by keywords and then taking random crops
  
    :param images:          A dataset containing a set of images in a similar 
                            format to that produced by 
                            torchvision.datasets.ImageFolder
                            In particular, image[i][0] should contain a torch 
                            tensor of shape [channels_i, height_i, width_i]
    :type images:           array-like
    :param masks:           A dataset containing a set of masks corresponding
                            to the set of images, in the same format as images
    :type masks:            array-like
    :param info:            A 2d sequence where the entries in the ith row 
                            contain descriptions of images[i]
                            info[i][0] should contain keywords to match for
                            that image
    :type info:             array-like
    :param sample_size:     The size of samples in the dataset, in the form 
                            [height, width]
    :type sample_size:     sequence[int, int]
    :param get_label:       A one-parameter function that takes in a row of 
                            info, say info[i], and returns the associated label
                            (the class of images[i])
    :type get_label:        function(array-like)
    :param keywords:        A list of keywords that are matched to the
                            descriptions of images in info [i][0]
    :type keywords:         array-like, containing strings
    :param samples_per_image: the number of samples to generate per image
    :type samples_per_image: int
    :param class_mapping:   a dictionary mapping class names to indices if
                            provided, otherwise one will be created
    :type class_mapping:    dict{ str: int } or None
    """

    def __init__(self, images, masks, info, sample_size, get_label,
                 samples_per_image=1, keywords=None, class_mapping=None):
        # create a list of the indices of the images that match the keywords
        if keywords is None:
            keywords = []
        to_select = list(range(len(info)))
        for keyword in keywords:
            to_select = [i for i in to_select if (keyword in info[i][0])]

        # create the class to index mapping
        if class_mapping is None:
            self.class_mapping = create_class_mapping(
                info[to_select], get_label)
        else:
            self.class_mapping = class_mapping
            # remove samples corresponding to a class not in the mapping
            to_select = [i for i in to_select if (
                    get_label(info[i]) in self.class_mapping)]

        self.out_channels = len(self.class_mapping)

        # generate random coordinates for the crop
        top_coords = np.array([], dtype=int)
        left_coords = np.array([], dtype=int)
        for i in to_select:
            image_size = images[i][0].shape[1:]
            max_top = image_size[0] - sample_size[0]
            max_left = image_size[1] - sample_size[1]

            if max_top < 0 or max_left < 0:
                print("Error: Image size", image_size,
                      "is smaller than sample size", sample_size)
                return

            top_coords = np.concatenate([top_coords, np.random.randint(
                0, max_top, size=samples_per_image)])
            left_coords = np.concatenate([left_coords, np.random.randint(
                0, max_left, size=samples_per_image)])

        # store images and masks
        self.images = torch.stack([
            TF.crop(images[to_select[i // samples_per_image]][0],
                    top_coords[i], left_coords[i], sample_size[0],
                    sample_size[1]
                    ) for i in range(len(top_coords))
        ])

        self.masks = torch.stack([
            create_mask(masks[to_select[i // samples_per_image]][0],
                        top_coords[i], left_coords[i], sample_size[0],
                        sample_size[1],
                        self.class_mapping[get_label(
                            info[to_select[i // samples_per_image]]
                        )],
                        ) for i in range(len(top_coords))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def tst_directory():
    """
    Provides a quick check for saving images from pytorch to a freshly
    made directory
    """

    name = 'SaveTest'  # name directory
    make_dir(name)  # make directory
    for j in range(20):
        # create a stack of 8 colored images of size 28 by 28 by 28
        x = Variable(torch.rand(8, 1, 28, 28))
        img_name = './' + name + '/image{}.png'.format(j)
        save_decoded_image2d(x, img_name)
    time.sleep(5)
    shutil.rmtree(name)


if __name__ == "__main__":
    tst_directory()
