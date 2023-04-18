import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from dlsia.core.networks import scale_up_down
from torch.autograd import Variable


class MSDNet2(nn.Module):
    """
    Defines Mixed Scale Dense Network based on topology (number of
    layers and channels per layer) and morphology (dilation sizes for
    each feature map). Input and output layer sizes are passed as
    input, including user-defined activation dropout rates and
    activation layers.

    :param int in_channels: number of channels in input data
    :param int out_channels: number of channels in output data
    :param num_layers: depth of network
    :type num_layers: int or None
    :param max_dilation: maximum dilation size the network will
                         cycle through
    :type max_dilation: int or None
    :param custom_msdnet: n by m numnpy array whose dimensions
                          define the network topology (m number of
                          layers and n channels per layer) and
                          entries define the morphology (size of
                          dilated kernel at each channel)
    :type custom_msdnet: List[int]
    :param activation: instance of PyTorch activation class applied
                       to each layer. If passing a list of
                       multiple activation class instances, each
                       will be applied in the order given.
                       ex) activation=nn.ReLU()
                       ex) activation=nn.RReLU(lower=.1, upper=.9)
                       ex) activation=[nn.ReLU(), nn.Sigmoid()]
    :type activation: torch.nn class instance or list of torch.nn
                      class instances
    :param normalization: PyTorch normalization class applied to
                          each layer. Passed as class without
                          parentheses since we need a different
                          instance per layer
                          ex) normalization=nn.BatchNorm2d
    :type normalization: torch.nn class
    :param final_layer: instance of PyTorch activation class applied
                        after final layer. If passing a list of
                        multiple activation class instances, each
                        will be applied in the order given.
                          ex) normalization=nn.Sigmoid()
                          ex) normalization=nn.Softmax(dim=1)
    :param int conv_kernel_size: the n by n size of filter mask applied to all
                                 but final convolution (3 or 5 chosen almost
                                 universally for CNNs)
    :param convolution: instance of PyTorch convolution class.
                        Accepted are nn.Conv1d, nn.Conv2d, and
                        nn.Conv3d.
    :type convolution: torch.nn class instance
    :param str padding_mode: padding mode to be used in convolution
                             to fill boundary space. Accepted input
                             are  'zeros', 'reflect', 'replicate' or
                             'circular'

    Instructions: User has two options for building network topology and
    morphology:
        1)  to generate a network based on number of layers, channels,
            and the number of dilations to cycle through, pass integers
            in for num_layers, layer_width, and max_dilation
            and leave custom_msdnet = None
                ex) num_layers=5, layer_width=2, max_dilation=5
                    yields the network [1, 3, 5, 2, 4, 1
                                        2, 4, 1, 3, 5, 2]
        2)  to create a custom network with user-specified dilation
            sizes, network depth, and layer width, pass in:
                    a) num_layers = layer_width
                                  = max_dilation
                                  = None
                    b)  numpy array populated with desired
                        integer-valued dilations whose columns
                        correspond to network layers and number of rows
                        corresponds to number of channels per layer
    Referenece paper: A mixed-scale dense convolutional neural network for
                      image analysis
    Published: PNAS, Jan. 2018
    Link: http://www.pnas.org/content/early/2017/12/21/1715832114

    Note: Bias=True is Conv2d and Conv3d default
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=10,
                 max_dilation=10,
                 custom_msdnet=None,
                 activation=nn.ReLU(),
                 normalization=nn.BatchNorm2d,
                 final_layer=None,
                 conv_kernel_size=3,
                 convolution=nn.Conv2d,
                 padding_mode="zeros"):
        """
        Build an MSDNet given the spoecs listed below.

        :param int in_channels: number of channels in input data
        :param int out_channels: number of channels in output data
        :param num_layers: depth of network
        :type num_layers: int or None
        :param max_dilation: maximum dilation size the network will
                             cycle through
        :type max_dilation: int or None
        :param custom_msdnet: n by m numnpy array whose dimensions
                              define the network topology (m number of
                              layers and n channels per layer) and
                              entries define the morphology (size of
                              dilated kernel at each channel)
        :type custom_msdnet: List[int]
        :param activation: instance of PyTorch activation class applied
                           to each layer. If passing a list of
                           multiple activation class instances, each
                           will be applied in the order given.
                           ex) activation=nn.ReLU()
                           ex) activation=nn.RReLU(lower=.1, upper=.9)
                           ex) activation=[nn.ReLU(), nn.Sigmoid()]
        :type activation: torch.nn class instance or list of torch.nn
                          class instances
        :param normalization: PyTorch normalization class applied to
                              each layer. Passed as class without
                              parentheses since we need a different
                              instance per layer
                              ex) normalization=nn.BatchNorm2d
        :type normalization: torch.nn class
        :param final_layer: instance of PyTorch activation class applied
                            after final layer. If passing a list of
                            multiple activation class instances, each
                            will be applied in the order given.
                              ex) normalization=nn.Sigmoid()
                              ex) normalization=nn.Softmax(dim=1)
        :param int conv_kernel_size: the n by n size of filter mask applied
                                to all but final convolution (3 or 5
                                chosen almost universally for CNNs)
        :param convolution: instance of PyTorch convolution class.
                            Accepted are nn.Conv1d, nn.Conv2d, and
                            nn.Conv3d.
        :type convolution: torch.nn class instance
        :param str padding_mode: padding mode to be used in convolution
                                 to fill boundary space. Accepted input
                                 are  'zeros', 'reflect', 'replicate' or
                                 'circular'
        """
        super(MSDNet2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.max_dilation = max_dilation
        self.custom_msdnet = custom_msdnet
        self.activation = activation
        self.normalization = normalization
        self.final_layer = final_layer
        self.conv_kernel_size = conv_kernel_size
        self.convolution = convolution
        self.padding_mode = padding_mode

        # Build dilations
        if self.custom_msdnet == None:
            tmp = np.arange(self.max_dilation) + 1
        else:
            tmp = custom_msdnet

        num_rep = int(np.ceil(self.num_layers / len(tmp)))

        self.dilations = np.tile(tmp, num_rep)
        self.dilations = self.dilations[0 : self.num_layers]

        self.connections = [[0 for i in range(self.num_layers+1)] for j in
                                 range(self.num_layers + 1)]

        #self.activations = [0 for i in range(self.num_layers + 1)]


        for out_layer in range(1,self.num_layers + 1):
            for in_layer in range(out_layer):


                tmp = self.build_conv_operator(self.dilations[out_layer-1])
                self.connections[in_layer][out_layer] = f"Connection" \
                                                        f"_{in_layer}_to_{out_layer}"
                self.add_module(self.connections[in_layer][out_layer], tmp)

            # Add activations, one per layer
            # self.activations[out_layer] = f"Activation_{out_layer}"
            #self.add_module(self.activation[out_layer], self.activation)

        # Add final convolution of kernel size 1
        self.connections[in_layer+1][out_layer] = "Final_Convolution"
        self.add_module(self.connections[in_layer+1][out_layer],
                        self.convolution(self.num_layers + 1,
                                         self.out_channels,
                                         kernel_size=1))




    def build_conv_operator(self, dil):
        """
        Build all convolutions connected to the current layer

        :param current_layer: the
        :param dilations
        :return: returns an operator
        """

        # Preallocate modules to house the operator
        modules = []

        dilated_conv_kernel_size = (self.conv_kernel_size - 1) * dil + 1
        padding = dilated_conv_kernel_size // 2

        # Add first convolution
        modules.append(self.convolution(1,1,
                                        kernel_size=self.conv_kernel_size,
                                        dilation=dil,
                                        padding=padding,
                                        padding_mode=self.padding_mode
                                        )
                       )

        # Finally, wrap all modules together in nn.Sequential
        operator = nn.Sequential(*modules)

        return operator

    def forward(self, x):
        """
        Default forward operator.

        :param x: input tensor.
        :return: output of neural network
        """

        x_main = x

        for out_layer in range(1, self.num_layers + 1):

            tmp = []

            for in_layer in range(out_layer):

                tmp.append(
                    self._modules[self.connections[in_layer][out_layer]](
                        x_main[:, in_layer, :, :].unsqueeze(1)))


            tmp = torch.cat(tmp, dim=1)

            if len(tmp.size()) == 3:
                tmp = torch.unsqueeze(tmp,1)
            else:
                tmp = torch.sum(tmp, dim=1)
                tmp = torch.unsqueeze(tmp, 1)
                tmp = self.activation(tmp)

            x_main = torch.cat((x_main, tmp), 1)



            #if out_layer == 3:
            #    assert 5==9


        # Apply final convolution for output layer

        x_out = self._modules[self.connections[in_layer+1][out_layer]](x_main)


        return x_out


    def topology_dict(self):
        """
        Get all parameters needed to build this network

        :return: An orderdict with all parameters needed
        :rtype: OrderedDict
        """

        topo_dict = OrderedDict()
        topo_dict["image_shape"] = self.image_shape
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_channels"] = self.out_channels
        topo_dict["depth"] = self.depth
        topo_dict["base_channels"] = self.base_channels
        topo_dict["growth_rate"] = self.growth_rate
        topo_dict["hidden_rate"] = self.hidden_rate
        topo_dict["carryover_channels"] = self.carryover_channels
        topo_dict["conv_kernel"] = self.conv_kernel
        topo_dict["kernel_down"] = self.kernel_down
        topo_dict["kernel_up"] = self.kernel_up
        topo_dict["normalization"] = self.normalization
        topo_dict["activation"] = self.activation
        topo_dict["conv_kernel_size"] = self.conv_kernel_size
        topo_dict["maxpool_kernel_size"] = self.maxpool_kernel_size
        topo_dict["stride"] = self.stride
        topo_dict["dilation"] = self.dilation
        return topo_dict

    def save_network_parameters(self, name=None):
        """
        Save the network parameters
        :param name: The filename
        :type name: str
        :return: None
        :rtype: None
        """
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


def MSDNet2_from_file(filename):
    """
    Construct an Autoencoder from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An Autoencoder
    :rtype: Autoencoder
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))
    msdnet2_obj = MSDNet2(**network_dict["topo_dict"])
    msdnet2_obj.load_state_dict(network_dict["state_dict"])
    return msdnet2_obj


def tst():
    a = 12
    b = 12
    custom_msdnet=[2,4,8,16,32]
    obj = MSDNet2(in_channels=1,
                  out_channels=1,
                  num_layers=10,
                  custom_msdnet=custom_msdnet
                  #max_dilation = 7
                  )

    print(obj)

    from torchsummary import summary
    from dlsia.core import helpers
    device = helpers.get_device()
    obj.to(device)
    summary(obj, (1, 64, 64))

    x = Variable(
        torch.rand(2, 1, a, b))
    x = x.cuda()
    y = obj(x)

    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    assert tst()
