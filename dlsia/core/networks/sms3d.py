import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from dlsia.core.networks import msd_graph_tools
from dlsia.core.networks import smsnet
from dlsia.core.networks import graph_utils
from dlsia.core.networks import scale_up_down
from torch.nn import Conv3d

def Conv3DReLU(in_channels,
               out_channels,
               conv_kernel_size,
               dilation_size,
               stride,
               output_padding,
               padding_mode="reflect"):
    """
    A simple Conv3D + ReLU operator

    :param in_channels: in channels
    :param out_channels: out channels
    :param conv_kernel_size: kernel size
    :param dilation_size: dilation size
    :param stride: Stride (for down sampling)
    :param output_padding: padding needed for transposed convolution.
    :param padding_mode: padding mode
    :return: a sequential module
    """

    # dilated_kernel_size = (conv_kernel_size - 1) * dilation_size + 1
    # padding = dilated_kernel_size // 2
    padding = int(dilation_size * (conv_kernel_size - 1) / 2)

    if output_padding[0] is None:

        function = nn.Sequential(Conv3d(in_channels, out_channels,
                                        kernel_size=conv_kernel_size,
                                        stride=stride,
                                        dilation=dilation_size,
                                        padding=padding,
                                        padding_mode=padding_mode),
                                 nn.BatchNorm3d(num_features=out_channels,
                                                track_running_stats=False),
                                 nn.ReLU()
                                 )
    else:
        function = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels,
                                                    kernel_size=conv_kernel_size,
                                                    stride=stride,
                                                    dilation=dilation_size,
                                                    padding=padding,
                                                    output_padding=output_padding,
                                                    padding_mode="zeros"),
                                 nn.BatchNorm3d(num_features=out_channels,
                                                track_running_stats=False),
                                 nn.ReLU(),
                                 )
    return function


def Linear3D(in_channels, out_channels, stride, output_padding, bias=True):
    """
    A linear transformation

    :param in_channels: in channels
    :param out_channels: out channels
    :param stride: stride
    :param output_padding: output padding for transpose convolution
    :param bias: refine a bias term?
    :return: an nn.sequential module
    """

    if output_padding[0] is None:
        function = nn.Sequential(Conv3d(in_channels, out_channels,
                                        kernel_size=1,
                                        dilation=1,
                                        padding=0,
                                        stride=stride,
                                        padding_mode="zeros", bias=bias)
                                 )
    else:
        function = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels,
                                                    kernel_size=1,
                                                    dilation=1,
                                                    padding=0,
                                                    stride=stride,
                                                    output_padding=output_padding,
                                                    padding_mode="zeros",
                                                    bias=bias))
    return function


def Linear3DSigmoid(in_channels, out_channels, stride, output_padding, bias=True):
    """
    A linear transformation

    :param output_padding: output padding needed for transpose convolutions
    :param bias: Refine a bias?
    :param stride: The stride
    :param in_channels: in channels
    :param out_channels: out channels
    :return: a sequential module
    """
    if output_padding[0] is None:
        function = nn.Sequential(Conv3d(in_channels, out_channels,
                                        kernel_size=1,
                                        dilation=1,
                                        padding=0,
                                        stride=stride,
                                        padding_mode="reflect",
                                        bias=bias),
                                 nn.Sigmoid())
    else:
        function = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels,
                                                    kernel_size=1,
                                                    dilation=1,
                                                    padding=0,
                                                    stride=stride,
                                                    output_padding=output_padding,
                                                    padding_mode="zeros",
                                                    bias=bias),
                                 nn.Sigmoid())
    return function


def Identity(in_channels, out_channels):
    """
    Identity operator.

    :param in_channels: not used
    :param out_channels: not used
    :return: the identity operator
    """
    function = nn.Identity()
    return function


def aggregate(node_layer_accumulator, this_node, data):
        """
        Aggregate the data in place
        :param node_layer_accumulator: a lit of lists
        :param this_node: where we place stuff
        :param data: the data to add
        """
        if node_layer_accumulator[this_node] is None:
            node_layer_accumulator[this_node] = data
        else:
            node_layer_accumulator[this_node] = \
                torch.cat([node_layer_accumulator[this_node], data], 1)
        return node_layer_accumulator



class SMSNet3D(nn.Module):
    """
    A Sparse Mixed Scale Network, 3D
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 in_shape,
                 out_shape,
                 scaling_table,
                 network_graph,
                 channel_count,
                 convolution_kernel_size=3,
                 first_action=Identity,
                 hidden_action=Conv3DReLU,
                 last_action=Linear3D,
                 ):
        """
        Build a network based on a specified graph.

        :param in_channels: in channels
        :param out_channels: out channels
        :param network_graph: a graph with properties
        :param channel_count: channel count object
        :param convolution_kernel_size: kernel size
        :param first_action: nn function that acts on data before it goes into
                             the network
        :param hidden_action: nn function for hidden layers
        :param last_action: nn function after last node
        """

        super(SMSNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.scaling_table = scaling_table

        self.network_graph = network_graph
        self.channel_count = channel_count

        self.first_action = first_action
        self.hidden_action = hidden_action
        self.last_action = last_action

        self.convolution_kernel_size = convolution_kernel_size

        self.edge_list, \
        self.source, \
        self.hidden, \
        self.sink = smsnet.edges_source_sink_list(self.network_graph)

        self.node_layer_accumulator = OrderedDict()
        self.action_list = []

        self.build_network()

        self.return_before_last_layer_ = False

    def build_network(self):
        """
        Put the network and its operations together
        """
        # the first thing to do is always the first_action
        # we choose tho keep the dimensions intact (for now)
        # self.add_module('on_input', self.first_action(self.in_channels,
        # self.in_channels))
        # self.action_list.append(('on_input', None, 0))

        # we want to know the last node

        last_node = next(reversed(self.channel_count))
        # now we traverse the network

        for edges in self.edge_list:
            self.node_layer_accumulator[edges[0][0]] = None

            for edge in edges:
                from_here = edge[0]
                to_there = edge[1]
                these_in_channels = self.channel_count[from_here]

                props = self.network_graph.get_edge_data(from_here, to_there)
                those_out_channels = props['channels']
                this_dilation = props['dilation']

                # get the sizes / powers
                from_size_power = self.network_graph.nodes[from_here]["size"]
                to_size_power = self.network_graph.nodes[to_there]["size"]

                scales_D3 = self.scaling_table["D3"][from_size_power][to_size_power]
                scales_W3 = self.scaling_table["W3"][from_size_power][to_size_power]
                scales_H3 = self.scaling_table["H3"][from_size_power][to_size_power]
                scales_D1 = self.scaling_table["D1"][from_size_power][to_size_power]
                scales_W1 = self.scaling_table["W1"][from_size_power][to_size_power]
                scales_H1 = self.scaling_table["H1"][from_size_power][to_size_power]

                if to_there == last_node:
                    self.add_module('Edge_%s_%s' % (from_here, to_there),
                                    Linear3D(these_in_channels,
                                             those_out_channels,
                                             stride=(scales_W1[0], scales_H1[0], scales_D1[0]),
                                             output_padding=(scales_W1[1], scales_H1[1], scales_D1[1]),
                                             bias=False))

                else:
                    self.add_module('Edge_%s_%s' % (from_here, to_there),
                                    self.hidden_action(these_in_channels,
                                                       those_out_channels,
                                                       self.convolution_kernel_size,
                                                       this_dilation,
                                                       stride=(scales_W3[0],
                                                               scales_H3[0],
                                                               scales_D3[0]),
                                                       output_padding=(scales_W3[1],
                                                                       scales_H3[1],
                                                                       scales_D3[1]
                                                                       )
                                                       ))
                what_now = ('Edge_%s_%s' % (from_here, to_there), from_here, to_there)
                self.action_list.append(what_now)

        last_node = next(reversed(self.channel_count))
        last_action_in = self.channel_count[last_node]
        last_action_out = self.out_channels

        self.action_list.append(('Last_Action', last_node, None))

        if self.last_action is None:
            self.add_module('Last_Action', nn.Identity())
        else:
            self.add_module('Last_Action', self.last_action(last_action_in,
                                                            last_action_out,
                                                            stride=(1, 1, 1),
                                                            output_padding=(None, None, None)))

    def forward(self, x):
        """
        The forward method

        :param x: The data
        :return: the resulting tensor after it has been passed through the network
        """

        node_layer_accumulator = OrderedDict()
        for node in self.channel_count:
            node_layer_accumulator[node] = None
        last_node = next(reversed(self.channel_count))
        node_layer_accumulator[last_node] = None

        aggregate(node_layer_accumulator, 0, x)

        for action_pair in self.action_list[:-1]:
            action, from_here, to_there = action_pair
            data_in = node_layer_accumulator[from_here]
            data_out = self._modules[action](data_in)
            aggregate(node_layer_accumulator, to_there, data_out)

        action, from_here, to_there = self.action_list[-1]
        data_in = node_layer_accumulator[from_here]
        data_out = self._modules[action](data_in)

        if self.return_before_last_layer_:
            return data_in, data_out
        return data_out

    def topology_dict(self):
        """
        Get all parameters needed to build this network

        :return: An orderdict with all parameters needed
        :rtype: OrderedDict
        """
        topo_dict = OrderedDict()
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_channels"] = self.out_channels
        topo_dict["in_shape"] = self.in_shape
        topo_dict["out_shape"] = self.out_shape
        topo_dict["scaling_table"] = self.scaling_table
        topo_dict["network_graph"] = self.network_graph
        topo_dict["channel_count"] = self.channel_count
        topo_dict["convolution_kernel_size"] = self.convolution_kernel_size
        topo_dict["first_action"] = self.first_action
        topo_dict["hidden_action"] = self.hidden_action
        topo_dict["last_action"] = self.last_action
        return topo_dict

    def save_network_parameters(self, name):
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
        torch.save(network_dict, name)


def random_3DSMS_network(in_channels,
                         out_channels,
                         layers,
                         dilation_choices,
                         in_shape=(64, 64, 64),
                         out_shape=(64, 64, 64),
                         hidden_out_channels=None,
                         layer_probabilities=None,
                         sizing_settings=None,
                         dilation_mode="Edges",
                         network_type="Regression",
                         ):
    # TODO: Fix documentation
    """
    Build a random sparse mixed scale network

    :param in_shape: input shape of tensor
    :type in_shape: tuple (Z,Y,X)
    :param out_shape: output shape
    :type out_shape: tuple (Z,Y,X)
    :param sizing_settings: Determines the scaling operations. Needs better documentation
    :type sizing_settings: FIX THIS
    :param in_channels: input channels
    :param out_channels: output channels
    :param layers: The  umber of hidden layers between input and output node
    :param dilation_choices: An array of dilation values to choose from
    :param hidden_out_channels: An array of output channel choices
    :param layer_probabilities: A dictionary that describes how the graph is constructed.
        LL_alpha: 0 uniform skips ; the higher this gets, the less likely it is a skip connection is 'lengthy'
        LL_gamma: Degree probability P(degree) \propto degree^-LL_gamma
        LL_max_degree: limits the maximal degree per node
        IL: probability of a connection between input and hidden layer
        LO: probability of a connection between hidden and output layer
        IO: bool, sets connections between input and output layer
    :param dilation_mode: If "Edges" dilations will be assigned at random.
                             "Nodes" all edges to the same node have equal dilation
                             "NodeCyclic" cyle through the list of dilations in a cyclic fashion, per node
    :param network_type: If "Regression" or "Classification" the final action is a linear operator.
                            "Regression_Sigmoid" does softmax as the final action
    :return: A network with above settings.
    """

    if sizing_settings is None:
        sizing_settings = {'stride_base': 2,
                           'min_power': 0,
                           'max_power': 0}
    if layer_probabilities is None:
        layer_probabilities = {'LL_alpha': 0.75,
                               'LL_gamma': 2.0,
                               'LL_max_degree': None,
                               'LL_min_degree': 1,
                               'IL': 0.25,
                               'LO': 0.25,
                               'IO': True}
    assert network_type in ["Regression", "Classification", "Regression_Positive", "Regression_Sigmoid"]

    if hidden_out_channels is None:
        hidden_out_channels = [1]

    obj = msd_graph_tools.RandomMultiSourceMultiSinkGraph(1, 1, layers, np.arange(5) + 1,
                                                          LL_alpha=layer_probabilities['LL_alpha'],
                                                          LL_gamma=layer_probabilities['LL_gamma'],
                                                          LL_min_degree=layer_probabilities['LL_min_degree'],
                                                          LL_max_degree=layer_probabilities['LL_max_degree'],
                                                          IL_target=layer_probabilities['IL'],
                                                          LO_target=layer_probabilities['LO'],
                                                          IO_target=layer_probabilities['IO'])
    G = obj.build_matrix(False)
    G = smsnet.sort_and_rename(G)

    # Assign dilations to each edge
    assert dilation_mode in ["Node", "NodeCyclic", "Edges"]
    G = smsnet.buildSMSdilations(G,
                                 dilation_choices=dilation_choices,
                                 mode=dilation_mode)
    # Assign output channels to each edge
    G, NodeCount = smsnet.buildSMSnodechannels(G,
                                               in_channels,
                                               in_channel_choices=[1],
                                               hidden_out_channel_choices=hidden_out_channels,
                                               p_hidden_out_channel_choices=None,
                                               p_in_channel_choices=None)
    # Assign sizes to nodes
    powers = range(sizing_settings['min_power'], sizing_settings['max_power'] + 1)
    G = graph_utils.assign_size(G, None)

    # fix first and last nodes
    first_node = 0
    last_node = len(NodeCount) - 1
    G.nodes[first_node]["size"] = 0
    G.nodes[last_node]["size"] = 0

    # now we need to build a scaling table that dictates convolutional settings
    scaling_table_H_k3 = scale_up_down.scaling_table(input_size=in_shape[1],
                                                     stride_base=sizing_settings["stride_base"],
                                                     max_power=sizing_settings["max_power"],
                                                     min_power=sizing_settings["min_power"],
                                                     kernel=3)
    scaling_table_W_k3 = scale_up_down.scaling_table(input_size=in_shape[0],
                                                     stride_base=sizing_settings["stride_base"],
                                                     max_power=sizing_settings["max_power"],
                                                     min_power=sizing_settings["min_power"],
                                                     kernel=3)
    scaling_table_D_k3 = scale_up_down.scaling_table(input_size=in_shape[2],
                                                     stride_base=sizing_settings["stride_base"],
                                                     max_power=sizing_settings["max_power"],
                                                     min_power=sizing_settings["min_power"],
                                                     kernel=3)

    scaling_table_D_k1 = scale_up_down.scaling_table(input_size=in_shape[2],
                                                     stride_base=sizing_settings["stride_base"],
                                                     max_power=sizing_settings["max_power"],
                                                     min_power=sizing_settings["min_power"],
                                                     kernel=1)

    scaling_table_H_k1 = scale_up_down.scaling_table(input_size=in_shape[1],
                                                     stride_base=sizing_settings["stride_base"],
                                                     max_power=sizing_settings["max_power"],
                                                     min_power=sizing_settings["min_power"],
                                                     kernel=1)
    scaling_table_W_k1 = scale_up_down.scaling_table(input_size=in_shape[0],
                                                     stride_base=sizing_settings["stride_base"],
                                                     max_power=sizing_settings["max_power"],
                                                     min_power=sizing_settings["min_power"],
                                                     kernel=1)

    scaling_table = {"D3": scaling_table_D_k3,
                     "W3": scaling_table_W_k3,
                     "H3": scaling_table_H_k3,
                     "D1": scaling_table_D_k1,
                     "W1": scaling_table_W_k1,
                     "H1": scaling_table_H_k1}

    last_action = Linear3D
    if network_type == "Regression":
        last_action = Linear3D

    if network_type == "Regression_Sigmoid":
        last_action = Linear3DSigmoid

    SMS_obj = SMSNet3D(in_channels=in_channels,
                       out_channels=out_channels,
                       in_shape=in_shape,
                       out_shape=out_shape,
                       scaling_table=scaling_table,
                       network_graph=G,
                       channel_count=NodeCount,
                       last_action=last_action)

    return SMS_obj


def SMSNetwork3D_from_file(filename):
    """
    Construct an SMSNet from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An SMSNet
    :rtype: smsnet
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))
    SMSObj = SMSNet3D(**network_dict["topo_dict"])
    SMSObj.load_state_dict(network_dict["state_dict"])
    return SMSObj
