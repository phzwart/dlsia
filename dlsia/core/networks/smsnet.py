import networkx as nx
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
from collections import OrderedDict
from dlsia.core.networks import msd_graph_tools
from dlsia.core.networks import graph_utils
from dlsia.core.networks import scale_up_down
from torch.nn import Conv2d


def sort_and_rename(G):
    """
    Topologoically sort and rename a graph

    :param G: The input network
    :return: The sorted network
    """
    to_be_removed = []
    for node_deg in G.degree:
        if node_deg[1] == 0:
            to_be_removed.append(node_deg[0])

    for node in to_be_removed:
        G.remove_node(node)

    rename = {}
    top_sort = list(nx.algorithms.dag.topological_sort(G))
    for ii, jj in enumerate(top_sort):
        rename[jj] = ii

    G = nx.relabel.relabel_nodes(G, rename)
    return G


def edges_source_sink_list(G):
    """
    Simple utility function to get a list of source, sink and edges

    :param G: input graph
    :return: edge_list, source_nodes, between_nodes, sink_nodes
    """

    edge_list = []

    sink_nodes = []
    source_nodes = []
    between_nodes = []

    for node in nx.nodes(G):
        these_edges = G.edges(node)

        if len(these_edges) > 0:
            edge_list.append(list(these_edges))

        N_out_degree = G.out_degree(node)
        N_in_degree = G.in_degree(node)

        if N_out_degree == 0:
            sink_nodes.append(node)
        if N_in_degree == 0:
            source_nodes.append(node)
        if N_out_degree > 0:
            if N_in_degree > 0:
                between_nodes.append(node)

    return edge_list, source_nodes, between_nodes, sink_nodes


def buildSMSdilations(G,
                      dilation_choices,
                      mode="Node",
                      p_dilation_choice=None,
                      p_sink_choice=None):
    """
    Assign dilations to the graph

    :param G: a digraph
    :param dilation_choices: dilation choices
    :param mode: Either "Node", "NodeCyclic", "Edges"
    :param p_dilation_choice: probabilities at which dilations are chosen
    :param p_sink_choice: probabilities at which dilations are chosen to the
                          sink
    :return: A digraph with listed dilations as edge properties
    """

    assert mode in ["Node", "NodeCyclic", "Edges"]
    dilation_choices = list(dilation_choices)
    sink_dilations = [1]
    edge_list, source, between, sink = edges_source_sink_list(G)

    if mode == "Node" or mode == "NodeCyclic":
        """
        Dilation choices are determined on the basis of the node in which the 
        edge lands in        
        """
        node_dilation_list = []
        if mode == "Node":
            node_dilation_list = np.random.choice(a=dilation_choices,
                                                  size=len(between),
                                                  replace=True,
                                                  p=p_dilation_choice)
        if mode == "NodeCyclic":
            reps = int(len(between) / len(dilation_choices) + .5)
            node_dilation_list = reps * dilation_choices
            node_dilation_list = node_dilation_list[0:len(between)]

        to_sink = np.random.choice(a=sink_dilations,
                                   size=len(sink),
                                   replace=True,
                                   p=p_sink_choice)

        node_dilation_list = np.hstack([len(source) * [0],
                                        node_dilation_list,
                                        to_sink])
        # now we loop over all edges and add weights
        for edges in edge_list:
            for edge in edges:
                to_there = edge[1]
                dilation_choice = node_dilation_list[to_there]
                G.add_edge(edge[0], edge[1], dilation=dilation_choice)

        return G

    if mode == "Edges":
        """
        Dilation choices are fully random for each edge
        apart from the connections to the sink.
        """

        for edges in edge_list:
            for edge in edges:
                to_there = edge[1]
                from_here = edge[0]

                if from_here == 0:
                    choices = sink_dilations
                    p = p_sink_choice
                else:
                    if to_there in sink:
                        choices = sink_dilations
                        p = p_sink_choice
                    else:
                        choices = dilation_choices
                        p = p_dilation_choice

                this_dilation = np.random.choice(a=choices,
                                                 size=1,
                                                 replace=True,
                                                 p=p)[0]
                G.add_edge(edge[0], edge[1], dilation=this_dilation)
        return G


def buildSMSnodechannels(G,
                         in_channels,
                         in_channel_choices=None,
                         hidden_out_channel_choices=None,
                         p_hidden_out_channel_choices=None,
                         p_in_channel_choices=None):
    """
    This function builds a distribution of input and output channels,
    in a random fashion given a SMS graph.


    :param G: input graph
    :param in_channels: the number of input channels
    :param in_channel_choices: governs number of channels from input to first
                               node
    :param hidden_out_channel_choices: governs number of output channels in
                                       hidden layers
    :param p_hidden_out_channel_choices: a distribution for the hidden channel
                                         choices
    :param p_in_channel_choices: a distribution for the in channel choices
    :return: An graph with the desired network parameters along the edges
    """
    if hidden_out_channel_choices is None:
        hidden_out_channel_choices = [1]

    # TODO: Fix this
    """ 
    The idea was this:   
    :param in_out_channels: The number of channels for the edge between the input to output node (if present).
                            By default, this is the same as the number of input channels.
    I ended up not using it. Will have to decide to add it in or not.
    """

    # first get some graph properties
    edge_list, source, hidden, sink = edges_source_sink_list(G)

    # An ordered Dict contains the aggregated layer count per node
    node_layer_count = OrderedDict()

    # we can do something more sophisticated,
    # but lets keep it simple
    for node in G.nodes():
        node_layer_count[node] = 0

    for edges in edge_list:
        for edge in edges:
            from_here = edge[0]
            to_here = edge[1]

            if from_here in source:
                node_layer_count[from_here] = in_channels
                if to_here in sink:
                    choices = [in_channels]
                    weights = [1.0]
                else:
                    choices = in_channel_choices
                    weights = p_in_channel_choices

            else:
                choices = hidden_out_channel_choices
                weights = p_hidden_out_channel_choices
            channels = np.random.choice(a=choices,
                                        size=1,
                                        replace=True,
                                        p=weights)[0]
            # add the channel
            G.add_edge(from_here, to_here, channels=channels)
            node_layer_count[to_here] += channels
    return G, node_layer_count


def Conv2DReLU(in_channels,
               out_channels,
               conv_kernel_size,
               dilation_size,
               stride,
               output_padding,
               padding_mode="reflect"):
    """
    A simple Conv2D + ReLU operator

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

        function = nn.Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size=conv_kernel_size,
                                        stride=stride,
                                        dilation=dilation_size,
                                        padding=padding,
                                        padding_mode=padding_mode),
                                 nn.BatchNorm2d(num_features=out_channels,
                                                track_running_stats=False),
                                 nn.ReLU()
                                 )
    else:
        function = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=conv_kernel_size,
                                                    stride=stride,
                                                    dilation=dilation_size,
                                                    padding=padding,
                                                    output_padding=output_padding,
                                                    padding_mode="zeros"),
                                 nn.BatchNorm2d(num_features=out_channels,
                                                track_running_stats=False),
                                 nn.ReLU(),
                                 )
    return function


def LinearReLU(in_channels, out_channels, stride, output_padding=None):
    """
    A linear transformation with subsequent ReLU

    :param in_channels: in channels
    :param out_channels: out channels
    :param stride: the stride
    :param output_padding: output padding needed to get image at right size. ignore is None.
    :return: a sequential module
    """
    if output_padding[0] is None:

        function = nn.Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size=1,
                                        dilation=1,
                                        padding=0,
                                        stride=stride,
                                        padding_mode="zeros"),
                                 nn.BatchNorm2d(num_features=out_channels,
                                                track_running_stats=False),
                                 nn.ReLU(), )
    else:
        function = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=1,
                                                    dilation=1,
                                                    padding=0,
                                                    stride=stride,
                                                    output_padding=output_padding,
                                                    padding_mode="zeros"),
                                 nn.BatchNorm2d(num_features=out_channels,
                                                track_running_stats=False),
                                 nn.ReLU()
                                 )
    return function


def Linear(in_channels, out_channels, stride, output_padding, bias=True):
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
        function = nn.Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size=1,
                                        dilation=1,
                                        padding=0,
                                        stride=stride,
                                        padding_mode="zeros", bias=bias))
    else:
        function = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=1,
                                                    dilation=1,
                                                    padding=0,
                                                    stride=stride,
                                                    output_padding=output_padding,
                                                    padding_mode="zeros",
                                                    bias=bias))
    return function


def LinearSigmoid(in_channels, out_channels, stride, output_padding, bias=True):
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
        function = nn.Sequential(Conv2d(in_channels, out_channels,
                                        kernel_size=1,
                                        dilation=1,
                                        padding=0,
                                        stride=stride,
                                        padding_mode="reflect",
                                        bias=bias),
                                 nn.Sigmoid())
    else:
        function = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
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


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        nn.init.orthogonal_(m.weight.data)  # , nonlinearity='relu')

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class SMSNet(nn.Module):
    """
    A Sparse Mixed Scale Network
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
                 hidden_action=Conv2DReLU,
                 last_action=Linear,
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

        super(SMSNet, self).__init__()

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
        self.sink = edges_source_sink_list(self.network_graph)

        self.node_layer_accumulator = OrderedDict()
        self.action_list = []
        for node in self.channel_count:
            self.node_layer_accumulator[node] = []

        self.build_network()

        self.return_before_last_layer_ = False

    def aggregate(self, this_node, data):
        """
        Aggregate the data in place

        :param this_node:
        :param data:
        """

        if self.node_layer_accumulator[this_node] is None:
            self.node_layer_accumulator[this_node] = data
        else:
            self.node_layer_accumulator[this_node] = \
                torch.cat([self.node_layer_accumulator[this_node], data], 1)

    def reset(self):
        for node in self.node_layer_accumulator:
            self.node_layer_accumulator[node] = None

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

                scales_W3 = self.scaling_table["W3"][from_size_power][to_size_power]
                scales_H3 = self.scaling_table["H3"][from_size_power][to_size_power]
                scales_W1 = self.scaling_table["W1"][from_size_power][to_size_power]
                scales_H1 = self.scaling_table["H1"][from_size_power][to_size_power]

                if from_here == 0:
                    self.add_module('Edge_%s_%s' % (from_here, to_there),
                                    Conv2DReLU(these_in_channels,
                                               those_out_channels,
                                               self.convolution_kernel_size,
                                               1,
                                                stride=(scales_W1[0], scales_H1[0]),
                                                output_padding=(scales_W1[1], scales_H1[1])
                                               ))
                else:
                    if to_there == last_node:
                        self.add_module('Edge_%s_%s' % (from_here, to_there),
                                        Linear(these_in_channels,
                                               those_out_channels,
                                               stride=(scales_W1[0], scales_H1[0]),
                                               output_padding=(scales_W1[1], scales_H1[1]),
                                               bias=False))

                    else:
                        self.add_module('Edge_%s_%s' % (from_here, to_there),
                                        self.hidden_action(these_in_channels,
                                                           those_out_channels,
                                                           self.convolution_kernel_size,
                                                           this_dilation,
                                                           stride=(scales_W3[0],
                                                                   scales_H3[0]),
                                                           output_padding=(scales_W3[1],
                                                                           scales_H3[1])
                                                           ))
                what_now = ('Edge_%s_%s' % (from_here, to_there), from_here, to_there)
                self.action_list.append(what_now)

        last_node = next(reversed(self.channel_count))
        last_action_in = self.channel_count[last_node]
        last_action_out = self.out_channels

        self.node_layer_accumulator[last_node] = None

        self.action_list.append(('Last_Action', last_node, None))

        if self.last_action is None:
            self.add_module('Last_Action', nn.Identity())
        else:
            self.add_module('Last_Action', self.last_action(last_action_in,
                                                            last_action_out,
                                                            stride=(1, 1),
                                                            output_padding=(None, None)))

    def forward(self, x):
        """
        The forward method

        :param x: The data
        :return: the resulting tensor after it has been passed through the network
        """
        self.reset()
        self.aggregate(0, x)

        for action_pair in self.action_list[:-1]:
            action, from_here, to_there = action_pair
            data_in = self.node_layer_accumulator[from_here]
            data_out = self._modules[action](data_in)
            self.aggregate(to_there, data_out)

        action, from_here, to_there = self.action_list[-1]

        data_in = self.node_layer_accumulator[from_here]

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


class SMS_reservoir(nn.Module):
    """
    An SMS Network but on the last layer is trained.
    """

    def __init__(self,
                 kitchen_sink_obj,
                 in_channels,
                 out_channels,
                 last_action=Linear
                 ):
        """
        Build a reservoir network.

        :param kitchen_sink_obj: An fixed SMS network
        :param in_channels: input channels
        :param out_channels: output channels
        :param last_action: what do we do in the end?
        """
        super(SMS_reservoir, self).__init__()

        self.kitchen_sink_obj = kitchen_sink_obj
        self.in_channels_in = in_channels
        self.out_channels = out_channels

        last_node = next(reversed(self.kitchen_sink_obj.channel_count))
        self.n_kitchen_sinks = self.kitchen_sink_obj.channel_count[last_node]
        self.last_action = last_action
        self.add_module('Last_Action', self.last_action(self.n_kitchen_sinks,
                                                        self.out_channels,
                                                        stride=(1, 1),
                                                        output_padding=(None, None)))

        self.return_before_last_layer_ = False

    def forward(self, x):
        """
        Standard forward method.

        :param x: input tensor
        :return: network output
        """
        with torch.no_grad():
            x_in = self.kitchen_sink_obj.eval()(x)
        x_out = self._modules["Last_Action"](x_in)
        if self.return_before_last_layer_:
            return x_in, x_out
        return x_out


def network_stats(net):
    order = net.network_graph.order()
    in_deg = net.network_graph.in_degree()
    out_deg = net.network_graph.out_degree()

    avg_in_deg = 0
    avg_out_deg = 0
    for ii in range(order):
        avg_in_deg += in_deg[ii]
        avg_out_deg += out_deg[ii]
    avg_in_deg /= order
    avg_out_deg /= order
    summary = {'order': order,
               'average_in_degree': avg_in_deg,
               'average_out_degree': avg_out_deg}
    return summary


def random_SMS_network(in_channels,
                       out_channels,
                       layers,
                       dilation_choices,
                       in_shape=(64, 64),
                       out_shape=(64, 64),
                       hidden_out_channels=None,
                       layer_probabilities=None,
                       sizing_settings=None,
                       dilation_mode="Edges",
                       network_type="Regression",
                       network_mode="Full"
                       ):
    """
    Build a random sparse mixed scale network

    :param in_channels: input channels
    :param out_channels: output channels
    :param in_shape: input shape - if no scaling /up down, this can be anything
    :param out_shape: output shape - if no scaling /up down, this can be anything
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
    :param sizing_settings: Governs sizing up and down nodes. Leave alone.
    :param dilation_mode: If "Edges" dilations will be assigned at random.
                             "Nodes" all edges to the same node have equal dilation
                             "NodeCyclic" cyle through the list of dilations in a cyclic fashion, per node
    :param network_type: If "Regression" or "Classification" the final action is a linear operator.
                            "Regression_Sigmoid" does softmax as the final action
    :param network_mode: "Full" train all parameters
                         "Reservoir" only train the final (linear) layer.
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
    assert network_mode in ["Full", "Reservoir"]

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
    G = sort_and_rename(G)

    # Assign dilations to each edge
    assert dilation_mode in ["Node", "NodeCyclic", "Edges"]
    G = buildSMSdilations(G,
                          dilation_choices=dilation_choices,
                          mode=dilation_mode)
    # Assign output channels to each edge
    in_channel_choices = [in_channels]
    #for this_one in hidden_out_channels:
    #    in_channel_choices.append(in_channels + this_one)

    G, NodeCount = buildSMSnodechannels(G,
                                        in_channels,
                                        in_channel_choices=in_channel_choices,
                                        hidden_out_channel_choices=hidden_out_channels,
                                        p_hidden_out_channel_choices=None,
                                        p_in_channel_choices=None)
    # Assign sizes to nodes
    powers = list(range(sizing_settings['min_power'], sizing_settings['max_power'] + 1))
    G = graph_utils.assign_size(G, powers)

    # fix first and last nodes, we are not doing any super resolution stuff yet
    first_node = 0
    last_node = len(NodeCount) - 1
    G.nodes[first_node]["size"] = 0
    size_ratio = -int(np.log(in_shape[0] / out_shape[0]) / np.log(sizing_settings['stride_base']))
    G.nodes[last_node]["size"] = size_ratio

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
    scaling_table = {"W3": scaling_table_W_k3,
                     "H3": scaling_table_H_k3,
                     "W1": scaling_table_W_k1,
                     "H1": scaling_table_H_k1}

    last_action = Linear
    if network_type == "Regression":
        last_action = Linear

    if network_type == "Regression_Sigmoid":
        last_action = LinearSigmoid

    if network_type == "Regression_Positive":
        last_action = LinearReLU

    if network_mode == "Full":
        SMS_obj = SMSNet(in_channels=in_channels,
                         out_channels=out_channels,
                         in_shape=in_shape,
                         out_shape=out_shape,
                         scaling_table=scaling_table,
                         network_graph=G,
                         channel_count=NodeCount,
                         last_action=last_action)
        return SMS_obj

    if network_mode == "Reservoir":
        SMS_kitchen_sink = SMSNet(in_channels=in_channels,
                                  out_channels=out_channels,
                                  in_shape=in_shape,
                                  out_shape=out_shape,
                                  scaling_table=scaling_table,
                                  network_graph=G,
                                  channel_count=NodeCount,
                                  last_action=None)
        SMS_kitchen_sink.requires_grad_(False)

        SMS_obj = SMS_reservoir(SMS_kitchen_sink,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                last_action=last_action
                                )

    return SMS_obj


def SMSNetwork_from_file(filename):
    """
    Construct an SMSNet from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An SMSNet
    :rtype: SMSNet
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))
    SMSObj = SMSNet(**network_dict["topo_dict"])
    SMSObj.load_state_dict(network_dict["state_dict"])
    return SMSObj


if __name__ == "__main__":
    tmp = random_SMS_network(in_channels=1,
                             out_channels=1,
                             in_shape=(64, 64),
                             out_shape=(64, 64),
                             layers=3,
                             dilation_choices=[1, 2, 4, 8],
                             hidden_out_channels=[1, 2, 4],
                             layer_probabilities={'LL_alpha': 0.50,
                                                  'LL_gamma': 3.0,
                                                  'LL_min_degree': 1,
                                                  'LL_max_degree': None,
                                                  'IL': 0.5,
                                                  'LO': 0.5,
                                                  'IO': True},
                             sizing_settings={'stride_base': 2,
                                              'min_power': -2,
                                              'max_power': 0},
                             dilation_mode="Edges",
                             network_type="Regression")
