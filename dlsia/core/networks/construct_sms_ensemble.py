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





















