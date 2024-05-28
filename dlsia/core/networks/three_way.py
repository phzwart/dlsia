import torch
import torch.nn as nn
import torch.nn.functional as F
from dlsia.core.networks.fcnet import FCNetwork
import random

class ThreeWay(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_network,
                 projection_layers = None,
                 dropout = 0.0,
                 clip_low = None,
                 clip_high = None,
                 final_action = None,
                 skip_connections=True
                 ):
        super(ThreeWay, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_network =  base_network
        self.intermediate_channels = self.base_network.out_channels
        self.dropout = dropout
        self.projection_layers = projection_layers
        self.final_action = final_action
        self.skip_connections = skip_connections

        if clip_low is not None:
            self.register_buffer('clip_low', clip_low)
        else:
            self.clip_low = None

        if clip_high is not None:
            self.register_buffer('clip_high', clip_high)
        else:
            self.clip_high = None

        if self.projection_layers is None:
            self.projection_layers = [ int(out_channels+self.intermediate_channels)//2 ]

        # projection head for the lower quantile
        tmp_in = 0
        if self.skip_connections:
            tmp_in = self.in_channels
        self.head_m1 = FCNetwork(self.intermediate_channels,
                                 self.projection_layers,
                                 self.out_channels,
                                 dropout_rate=dropout,
                                 skip_connections=self.skip_connections,
                                 o_channels = tmp_in
                                 )

        # projection head for the median
        self.head_0 = FCNetwork(self.intermediate_channels,
                                 self.projection_layers,
                                 self.out_channels,
                                 dropout_rate=dropout,
                                 skip_connections=self.skip_connections,
                                 o_channels = tmp_in)

        # projection head for the upper quantile
        self.head_p1 = FCNetwork(self.intermediate_channels,
                                 self.projection_layers,
                                 self.out_channels,
                                 dropout_rate=dropout,
                                 skip_connections=self.skip_connections,
                                 o_channels = tmp_in)

    def forward(self, o):
        x = self.base_network(o)
        if self.skip_connections:
            pass
        else:
            o = None

        lower_quantile_adjustment = F.softplus(self.head_m1(x,o))
        median = self.head_0(x,o)
        upper_quantile_adjustement = F.softplus(self.head_p1(x,o))

        lower_quantile = median - lower_quantile_adjustment
        upper_quantile = median + upper_quantile_adjustement

        if self.clip_low is not None:
            lower_quantile = torch.max(lower_quantile, self.clip_low)
            median = torch.max(median, self.clip_low)
            upper_quantile = torch.max(upper_quantile, self.clip_low)

        if self.clip_high is not None:
            lower_quantile = torch.min(lower_quantile, self.clip_high)
            median = torch.min(median, self.clip_high)
            upper_quantile = torch.min(upper_quantile, self.clip_high)

        if self.final_action is not None:
            lower_quantile = self.final_action(lower_quantile)
            median = self.final_action(median)
            upper_quantile = self.final_action(upper_quantile)

        return lower_quantile, median, upper_quantile

    def save_network_parameters(self, name=None):
        """
        Save the network parameters
        :param name: The filename
        :type name: str
        :return: None
        :rtype: None
        """
        network_dict = OrderedDict()
        network_dict["base_network"] = self.squash_head.topology_dict()
        network_dict["head_m1"] = self.squash_head.topology_dict()
        network_dict["head_0"] = self.squash_head.topology_dict()
        network_dict["head_p1"] = self.squash_head.topology_dict()

        if name is None:
            return network_dict
        torch.save(network_dict, name)


class RandomizedPinballLoss(nn.Module):
    def __init__(self, quantiles, biases=None, channel_weights=None):
        """ Initialize the Randomized Pinball Loss module.

        Args:
            quantiles (list of floats): Quantiles for which to calculate the pinball loss (e.g., [0.25, 0.5, 0.75]).
            biases (list of floats, optional): Relative likelihoods of selecting each quantile. If not provided, each quantile has equal chance.
        """
        super(RandomizedPinballLoss, self).__init__()
        assert len(quantiles) == len(biases) if biases else True, "Length of quantiles must match length of biases"
        self.quantiles = quantiles
        self.biases = biases if biases else [1] * len(quantiles)  # Equal probability if no biases are provided
        self.channel_weights = None

    def forward(self, predictions, y_true):
        """ Calculate the randomized pinball loss.

        Args:
            y_true (Tensor): The true values.
            predictions (list of Tensors): Predictions corresponding to each quantile.

        Returns:
            Tensor: The pinball loss for a randomly selected quantile.
        """
        # Normalize biases to get probabilities
        total_bias = sum(self.biases)
        probabilities = [b / total_bias for b in self.biases]

        # Select a quantile based on the given probabilities
        index = random.choices(range(len(self.quantiles)), weights=probabilities, k=1)[0]
        selected_quantile = self.quantiles[index]
        selected_prediction = predictions[index]

        # Compute pinball loss for the selected quantile
        return self.pinball_loss(y_true, selected_prediction, selected_quantile, self.channel_weights)

    @staticmethod
    def pinball_loss(y_true, y_pred, tau, channel_weights=None):
        """ Calculate the pinball loss for a single quantile.

        Args:
            y_true (Tensor): The true values.
            y_pred (Tensor): The predicted values.
            tau (float): The quantile to calculate the loss for.

        Returns:
            Tensor: The pinball loss.
        """
        errors = y_true - y_pred
        if channel_weights is not None:
            errors = errors * channel_weights
        return torch.mean(torch.maximum(tau * errors, (tau - 1) * errors))





