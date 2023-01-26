import torch
import torch.nn as nn
from collections import OrderedDict


class AggregateNet(nn.Module):
    """
    Final 1x1 convolutional layer that can be used to combine the results
    of multiple individual models

    :param int in_channels: number of channels in the input
    :param int out_channels: number of channels in the output
    :param bool relu: perform ReLU after the layer
    :param bool bias: include a bias parameter in the convolution layer

    """

    def __init__(self, in_channels, out_channels, layers=1, relu=True, bias=True, final_activation=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.relu = relu
        self.bias = bias
        self.final_activation = final_activation

        self.conv = []
        for ii in range(self.layers - 1):
            self.conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias))
            if relu:
                self.conv.append(nn.ReLU())
        self.conv = nn.Sequential(*self.conv)
        self.final_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, bias=bias)

    def forward(self, x):
        if self.layers > 1:
            x = self.conv(x)
        x = self.final_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def save_network_parameters(self, name):
        """
        Save an ordered dictionary of the state and topology parameters of the network.
        Can be used to save / reinstantiate the network.

        :param name: filename
        :type name: str
        :return: None
        :rtype: None
        """
        topo_dict = OrderedDict()
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_channels"] = self.out_channels
        topo_dict["layers"] = self.layers
        topo_dict["relu"] = self.relu
        topo_dict["bias"] = self.bias
        topo_dict["final_activation"] = self.final_activation
        state_dict = self.state_dict()
        network_params = OrderedDict()
        network_params["topo_dict"] = topo_dict
        network_params["state_dict"] = state_dict
        torch.save(network_params, name)


def AggNet_from_file(filename):
    """
    Read network parameter file from disc and reinstantiate it.

    :param filename: The filename
    :type filename: str
    :return: The Aggnet of interest
    :rtype: AggregateNet
    """
    network_params = torch.load(filename)
    obj = AggregateNet(**network_params["topo_dict"])
    obj.load_state_dict(network_params["state_dict"])
    return obj
