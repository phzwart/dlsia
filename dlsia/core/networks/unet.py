import numpy as np

import torch
import torch.nn as nn


class FixedSizeDoubleConv(nn.Module):
    """
    Two 3x3 convolution layers which preserve the size of the image, with 
    batchnorm and ReLU

    :param int in_channels:  # of channels in the input image
    :param int mid_channels: # of channels in the intermediate representation
                         (output of first convolution and input of second
                         convolution)
    :param int out_channels: # of channels in the output map
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = out_channels if mid_channels is None else mid_channels
        self.out_channels = out_channels

        self.fixed_size_double_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels, self.out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fixed_size_double_conv(x)


class UNet(nn.Module):
    """ 
    Implementation of a UNet (arXiv:1505.04597) with batchnorm and ReLU.
    MaxPool for downscaling and ConvTranspose for upscaling.


    :param int in_channels: # of channels in the input image
    :param int out_channels: # of channels in the output map
    :param int depth: The depth of the UNet, which is the number of
                      downscaling/upscaling operations + 1. Default depth of 5
                      is the depth used in the original paper
    :param int first_channels: # of channels in the first hidden layer. This
                      determines the # of channels in subsequent hidden layers
                      as the # of channels doubles after each double
                      convolution when downscaling. Default of 64 is the one
                      used in the original paper.
    """

    def __init__(self, in_channels, out_channels, depth=5, first_channels=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.first_channels = first_channels

        self.channels = np.array([in_channels])
        temp = np.array([first_channels * (2 ** i) for i in range(depth)])
        self.channels = np.concatenate([self.channels, temp])
        temp = np.flip(temp)
        self.channels = np.concatenate([self.channels, temp[1:]])
        # self.channels = [in_channels, first_channels, first_channels * 2, ..., 
        #       first_channels * 2^(depth-2), first_channels * 2^(depth-1), ..., 
        #       first_channels * 2, first_channels]

        self.convs = [FixedSizeDoubleConv(self.channels[i], self.channels[i + 1]
                                          ) for i in range(len(self.channels) - 1)]
        self.convs = nn.ModuleList(self.convs)

        self.downs = [nn.MaxPool2d(2) for _ in range(depth - 1)]
        self.downs = nn.ModuleList(self.downs)

        self.ups = [nn.ConvTranspose2d(temp[i], temp[i + 1],
                                       kernel_size=2, stride=2
                                       ) for i in range(depth - 1)]
        self.ups = nn.ModuleList(self.ups)

        self.final_layer = nn.Conv2d(first_channels, out_channels,
                                     kernel_size=1)

    def forward(self, x):
        down_intermediates = []

        # contracting path
        for i in range(self.depth - 1):
            x = self.convs[i](x)
            down_intermediates.append(x)
            x = self.downs[i](x)

        x = self.convs[self.depth - 1](x)

        # expanding path
        for i in range(self.depth - 1):
            x = self.ups[i](x)

            # traverse down_intermediates in reverse order
            x = torch.cat([x, down_intermediates[-(i + 1)]], dim=1)
            x = self.convs[self.depth + i](x)

        # final 1x1 convolution
        x = self.final_layer(x)
        return x


def tst():
    """
    TODO: Insert some test functionality here
    """

    print('dummy')


if __name__ == "__main__":
    tst()
