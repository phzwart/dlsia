import torch
from dlsia.core.networks import sms3d, smsnet
from dlsia.core import helpers


def construct_2dsms_ensembler(n_networks,
                              in_channels,
                              out_channels,
                           layers,
                           alpha = 0.0,
                           gamma = 0.0,
                           hidden_channels = None,
                           dilation_choices = [1,2,3,4],
                           P_IL = 0.995,
                           P_LO = 0.995,
                           P_IO = True,
                           parameter_bounds = None,
                           max_trial=100,
                           network_type="Regression",
                           parameter_counts_only = False
                           ):
    """
    Constructs an ensemble of 2D SMS (Sparse Multiscale) networks.

    Parameters:
    n_networks (int): The number of networks to include in the ensemble.
    in_channels (int): The number of input channels for the networks.
    out_channels (int): The number of output channels for the networks.
    layers (int): The number of layers for each network.
    alpha (float, optional): Alpha parameter for the LL (lower layer) probabilities. Default is 0.0.
    gamma (float, optional): Gamma parameter for the LL probabilities. Default is 0.0.
    hidden_channels (list, optional): The number of hidden channels for each layer. Defaults to [3 * out_channels].
    dilation_choices (list, optional): The choices of dilation rates. Defaults to [1, 2, 3, 4].
    P_IL (float, optional): Probability of an internal layer. Default is 0.995.
    P_LO (float, optional): Probability of a layer being the last one. Default is 0.995.
    P_IO (bool, optional): Whether the input/output connections are enabled. Default is True.
    parameter_bounds (tuple, optional): The minimum and maximum parameter bounds for networks. Default is None.
    max_trial (int, optional): The maximum number of trials to generate a network. Default is 100.
    network_type (str, optional): The type of network ('Regression' or other types). Default is 'Regression'.
    parameter_counts_only (bool, optional): Whether to return only the parameter counts instead of the network objects. Default is False.

    Returns:
    list: A list of constructed networks or their parameter counts if parameter_counts_only is True.
    """

    networks = []

    layer_probabilities = {
        'LL_alpha': alpha,
        'LL_gamma': gamma,
        'LL_max_degree': layers,
        'LL_min_degree': 1,
        'IL': P_IL,
        'LO': P_LO,
        'IO': P_IO,
    }


    if parameter_counts_only:
        assert parameter_bounds is None

    if hidden_channels is None:
        hidden_channels = [ 3*out_channels ]

    for _ in range(n_networks):
        ok = False
        count = 0
        while not ok:
            count += 1
            this_net = smsnet.random_SMS_network(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    layers=layers,
                                                    dilation_choices=dilation_choices,
                                                    hidden_out_channels=hidden_channels,
                                                    layer_probabilities=layer_probabilities,
                                                    sizing_settings=None,
                                                    dilation_mode="Edges",
                                                    network_type=network_type,
                                                    )
            pcount = helpers.count_parameters(this_net)
            if parameter_bounds is not None:
                if pcount > min(parameter_bounds):
                    if pcount < max(parameter_bounds):
                        ok = True
                        networks.append(this_net)
                if count > max_trial:
                    print("Could not generate network, check bounds")
            else:
                ok = True
                if parameter_counts_only:
                    networks.append(pcount)
                else:
                    networks.append(this_net)
    return networks


def construct_3dsms_ensembler(n_networks,
                              in_channels,
                              out_channels,
                           layers,
                           alpha = 0.0,
                           gamma = 0.0,
                           hidden_channels = None,
                           dilation_choices = [1,2,3,4],
                           P_IL = 0.995,
                           P_LO = 0.995,
                           P_IO = True,
                           parameter_bounds = None,
                           max_trial=100,
                           network_type="Regression",
                           parameter_counts_only = False
                           ):
    """
    Constructs an ensemble of 3D SMS (Sparse Multiscale) networks.

    Parameters:
    n_networks (int): The number of networks to include in the ensemble.
    in_channels (int): The number of input channels for the networks.
    out_channels (int): The number of output channels for the networks.
    layers (int): The number of layers for each network.
    alpha (float, optional): Alpha parameter for the LL (lower layer) probabilities. Default is 0.0.
    gamma (float, optional): Gamma parameter for the LL probabilities. Default is 0.0.
    hidden_channels (list, optional): The number of hidden channels for each layer. Defaults to [3 * out_channels].
    dilation_choices (list, optional): The choices of dilation rates. Defaults to [1, 2, 3, 4].
    P_IL (float, optional): Probability of an internal layer. Default is 0.995.
    P_LO (float, optional): Probability of a layer being the last one. Default is 0.995.
    P_IO (bool, optional): Whether the input/output connections are enabled. Default is True.
    parameter_bounds (tuple, optional): The minimum and maximum parameter bounds for networks. Default is None.
    max_trial (int, optional): The maximum number of trials to generate a network. Default is 100.
    network_type (str, optional): The type of network ('Regression' or other types). Default is 'Regression'.
    parameter_counts_only (bool, optional): Whether to return only the parameter counts instead of the network objects. Default is False.

    Returns:
    list: A list of constructed networks or their parameter counts if parameter_counts_only is True.
    """
                             
    networks = []

    layer_probabilities = {
        'LL_alpha': alpha,
        'LL_gamma': gamma,
        'LL_max_degree': layers,
        'LL_min_degree': 1,
        'IL': P_IL,
        'LO': P_LO,
        'IO': P_IO,
    }


    if parameter_counts_only:
        assert parameter_bounds is None

    if hidden_channels is None:
        hidden_channels = [ 3*out_channels ]

    for _ in range(n_networks):
        ok = False
        count = 0
        while not ok:
            count += 1
            this_net = sms3d.random_3DSMS_network(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    layers=layers,
                                                    dilation_choices=dilation_choices,
                                                    hidden_out_channels=hidden_channels,
                                                    layer_probabilities=layer_probabilities,
                                                    sizing_settings=None,
                                                    dilation_mode="Edges",
                                                    network_type=network_type,
                                                    )
            pcount = helpers.count_parameters(this_net)
            if parameter_bounds is not None:
                if pcount > min(parameter_bounds):
                    if pcount < max(parameter_bounds):
                        ok = True
                        networks.append(this_net)
                if count > max_trial:
                    print("Could not generate network, check bounds")
            else:
                ok = True
                if parameter_counts_only:
                    networks.append(pcount)
                else:
                    networks.append(this_net)
    return networks





















