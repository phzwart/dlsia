"""
Module: Cerberus Quantile Regression

This module contains the implementation of the Cerberus network, which is
designed for quantile regression. It outputs lower quantile, median, and upper
quantile predictions for input data, while offering various architectural
features like dropout, skip connections, and feature projection.

The `RandomizedPinballLoss` class interacts with the Cerberus network by
calculating the pinball loss for a randomly selected quantile prediction
(lower, median, or upper quantile). This stochastic approach helps balance
quantile predictions during training.

Additionally, the `PartialFeatureProjector` class provides functionality for
normalizing and projecting extracted features from the base network of
Cerberus. This allows further transformations or additional tasks beyond
quantile regression.

Classes:
- Cerberus: The main network for quantile regression.
- RandomizedPinballLoss: Loss function that randomly selects a quantile for
 pinball loss computation.
- PartialFeatureProjector: A feature projector that normalizes and projects
 features from the base network.

"""


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlsia.core.networks import smsnet, fcnet, tunet
from collections import OrderedDict


class RandomizedPinballLoss(nn.Module):
    def __init__(self, quantiles, biases=None, channel_weights=None):
        """ Initialize the Randomized Pinball Loss module.

        Args: quantiles (list of floats): Quantiles for which to calculate
        the pinball loss (e.g., [0.25, 0.5, 0.75]). biases (list of floats,
        optional): Relative likelihoods of selecting each quantile. If not
        provided, each quantile has equal chance.
        """
        super(RandomizedPinballLoss, self).__init__()
        assert len(quantiles) == len(
            biases) if biases else True, ("Length of quantiles must match "
                                          "length of biases")
        self.quantiles = quantiles
        self.biases = biases if biases else [1] * len(
            quantiles)  # Equal probability if no biases are provided
        self.channel_weights = channel_weights

    def forward(self, predictions, y_true):
        """ Calculate the randomized pinball loss.

        Args:
            y_true (Tensor): The true values.
            predictions (list of Tensors): Predictions corresponding to
            each quantile.

        Returns:
            Tensor: The pinball loss for a randomly selected quantile.
        """
        # Normalize biases to get probabilities
        total_bias = sum(self.biases)
        probabilities = [b / total_bias for b in self.biases]

        # Select a quantile based on the given probabilities
        index = random.choices(range(len(self.quantiles)),
                               weights=probabilities, k=1)[0]
        selected_quantile = self.quantiles[index]
        selected_prediction = predictions[index]

        # Compute pinball loss for the selected quantile
        return self.pinball_loss(y_true, selected_prediction,
                                 selected_quantile, self.channel_weights)

    @staticmethod
    def pinball_loss(y_true, y_pred, tau, channel_weights=None):
        """ Calculate the pinball loss for a single quantile.

        Args:
            y_true (Tensor): The true values.
            y_pred (Tensor): The predicted values.
            tau (float): The quantile to calculate the loss for.
            channel_weights: Tensor containing channel weights (optional)
        Returns:
            Tensor: The pinball loss.
        """
        errors = y_true - y_pred
        if channel_weights is not None:
            errors = errors * channel_weights
        return torch.mean(torch.maximum(tau * errors, (tau - 1) * errors))


class PartialFeatureProjector(nn.Module):
    """
    A class that normalizes input data and projects it into a different
    feature space using a learned projector matrix.

    Attributes:
    -----------
    mean : torch.Tensor
        The mean tensor used for normalization.
    sigma : torch.Tensor
        The standard deviation tensor used for normalization.
    projector : torch.Tensor
        The learned projector matrix to project features into a different
        space.
    """

    def __init__(self, mean, sigma, projector):
        """
        Initializes the PartialFeatureProjector with given mean, sigma,
        and projector tensors.

        Parameters:
        -----------
        mean : torch.Tensor
            The mean tensor for feature normalization.
        sigma : torch.Tensor
            The standard deviation tensor for feature normalization.
        projector : torch.Tensor
            The projector matrix used to transform features.
        """
        super(PartialFeatureProjector, self).__init__()
        self.register_buffer('mean', mean.clone().detach())
        self.register_buffer('sigma', sigma.clone().detach())
        self.register_buffer('projector', projector.clone().detach())


    def to(self, device):
        """
        Moves all model parameters to the specified device (CPU or GPU).

        Parameters:
        -----------
        device : torch.device
            The device to which the parameters should be moved.
        """
        self.mean = self.mean.to(device)
        self.sigma = self.sigma.to(device)
        self.projector = self.projector.to(device)

    def __call__(self, data):
        """
        Applies the feature normalization and projection to the input data.

        Parameters:
        -----------
        data : torch.Tensor
            Input tensor with shape (batch_size, num_channels, height, width).

        Returns:
        --------
        torch.Tensor
            Projected feature tensor after normalization.
        """
        assert len(
            data.shape) == 4, ("Input data should be of shape (batch_size, "
                               "num_channels, height, width)")
        projected_data = (data - self.mean[None, :, None, None]) / (
            self.sigma[None, :, None, None])
        projected_data = torch.einsum('nk... , kl -> nl...',
                                      projected_data,
                                      self.projector)
        return projected_data

    def save_parameters(self):
        """
        Saves the mean, sigma, and projector parameters as a dictionary.

        Returns:
        --------
        OrderedDict
            A dictionary containing mean, sigma, and projector tensors moved
            to CPU.
        """
        param_dict = OrderedDict()
        param_dict["mean"] = self.mean.cpu()
        param_dict["sigma"] = self.sigma.cpu()
        param_dict["projector"] = self.projector.cpu()
        return param_dict


def PartialFeatureProjector_from_file(filename):
    """
    Loads the PartialFeatureProjector parameters from a file or OrderedDict.

    Parameters:
    -----------
    filename : str or OrderedDict
        The file path to load the parameters from, or an OrderedDict
        containing the parameters.

    Returns:
    --------
    PartialFeatureProjector
        An instance of the PartialFeatureProjector class initialized with
        loaded parameters.
    """
    if isinstance(filename, OrderedDict):
        param_dict = filename
    else:
        param_dict = torch.load(filename, map_location=torch.device('cpu'))

    result = PartialFeatureProjector(**param_dict)
    return result


class Cerberus(nn.Module):
    """
    A neural network model designed to estimate lower quantile, median,
    and upper quantile outputs
    for a given input. It includes options for dropout, projection layers,
    feature clipping, skip
    connections, and partial feature projection.

    Attributes:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    base_network : nn.Module
        Base network used for feature extraction.
    net_lower_quantile : nn.Module or None
        Network responsible for estimating the lower quantile.
    net_median : nn.Module or None
        Network responsible for estimating the median.
    net_upper_quantile : nn.Module or None
        Network responsible for estimating the upper quantile.
    projection_layers : list of int or None
        Projection layers for feature transformation.
    dropout : float
        Dropout rate for regularization.
    clip_low : torch.Tensor or None
        Tensor that clips the lower bound of the outputs.
    clip_high : torch.Tensor or None
        Tensor that clips the upper bound of the outputs.
    final_action : callable or None
        Optional final function applied to the quantile outputs.
    skip_connections : bool
        Whether to use skip connections from input to output.
    partial_feature_projector : callable or None
        Optional partial feature projector applied to extracted features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_network,
                 net_lower_quantile=None,
                 net_median=None,
                 net_upper_quantile=None,
                 projection_layers=None,
                 dropout=0.0,
                 clip_low=None,
                 clip_high=None,
                 final_action=None,
                 skip_connections=True,
                 partial_feature_projector=None
                 ):
        """
        Initializes the Cerberus model with specified input parameters.

        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        base_network : nn.Module
            Base feature extractor network.
        net_lower_quantile : nn.Module, optional
            Network for lower quantile projection, defaults to None.
        net_median : nn.Module, optional
            Network for median projection, defaults to None.
        net_upper_quantile : nn.Module, optional
            Network for upper quantile projection, defaults to None.
        projection_layers : list of int, optional
            Layers used for projecting features, defaults to None.
        dropout : float, optional
            Dropout rate, defaults to 0.0.
        clip_low : torch.Tensor, optional
            Minimum value to clip the output, defaults to None.
        clip_high : torch.Tensor, optional
            Maximum value to clip the output, defaults to None.
        final_action : callable, optional
            Optional function to apply to final outputs, defaults to None.
        skip_connections : bool, optional
            Whether to use skip connections from input to output, defaults
            to True.
        partial_feature_projector : callable, optional
            Optional partial feature projector, defaults to None.
        """
        super(Cerberus, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_network = base_network
        self.intermediate_channels = self.base_network.out_channels
        self.dropout = dropout
        self.projection_layers = projection_layers
        self.final_action = final_action
        self.skip_connections = skip_connections
        self.partial_feature_projector = partial_feature_projector

        if self.partial_feature_projector is not None:
            self.init_feature_projector(partial_feature_projector)
            self.register_buffer('partial_feature_mean',
                                self.partial_feature_projector.mean)
            self.register_buffer('partial_feature_sigma',
                                self.partial_feature_projector.sigma)
            self.register_buffer('partial_feature_projector_matrix',
                                self.partial_feature_projector.projector)
        if clip_low is not None:
            self.register_buffer('clip_low', clip_low)
        else:
            self.clip_low = None

        if clip_high is not None:
            self.register_buffer('clip_high', clip_high)
        else:
            self.clip_high = None

        if self.projection_layers is None:
            if net_lower_quantile is None:
                self.projection_layers = [
                    int(out_channels + self.intermediate_channels) // 2]
            else:
                self.projection_layers = net_lower_quantile.Cmiddle

        tmp_in = 0
        if self.skip_connections:
            tmp_in = self.in_channels

        # Projection head for the lower quantile
        self.head_m1 = net_lower_quantile
        if net_lower_quantile is None:
            self.head_m1 = fcnet.FCNetwork(
                self.intermediate_channels,
                self.projection_layers,
                self.out_channels,
                dropout_rate=dropout,
                skip_connections=self.skip_connections,
                o_channels=tmp_in)

        # Projection head for the median
        self.head_0 = net_median
        if self.head_0 is None:
            self.head_0 = fcnet.FCNetwork(
                self.intermediate_channels,
                self.projection_layers,
                self.out_channels,
                dropout_rate=dropout,
                skip_connections=self.skip_connections,
                o_channels=tmp_in)

        # Projection head for the upper quantile
        self.head_p1 = net_upper_quantile
        if net_upper_quantile is None:
            self.head_p1 = fcnet.FCNetwork(
                self.intermediate_channels,
                self.projection_layers,
                self.out_channels,
                dropout_rate=dropout,
                skip_connections=self.skip_connections,
                o_channels=tmp_in)

    def forward(self, o):
        """
        Forward pass through the Cerberus network.

        Parameters:
        -----------
        o : torch.Tensor
            Input tensor.

        Returns:
        --------
        tuple of torch.Tensor
            Lower quantile, median, upper quantile, and optional partial
            features if available.
        """
        x = self.base_network(o)
        if self.skip_connections:
            pass
        else:
            o = None

        lower_quantile_adjustment = F.softplus(self.head_m1(x, o))
        median = self.head_0(x, o)
        upper_quantile_adjustement = F.softplus(self.head_p1(x, o))

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

        if self.partial_feature_projector is not None:
            partial_features = self.partial_feature_projector(x)
            return lower_quantile, median, upper_quantile, partial_features

        return lower_quantile, median, upper_quantile

    def init_feature_projector(self, feature_projector):
        self.partial_feature_projector = feature_projector

        self.register_buffer('partial_feature_mean',
                            self.partial_feature_projector.mean)
        self.register_buffer('partial_feature_sigma',
                            self.partial_feature_projector.sigma)
        self.register_buffer('partial_feature_projector_matrix',
                            self.partial_feature_projector.projector)


    def save_network_parameters(self, name=None):
        """
        Save the network parameters to a file.

        Parameters:
        -----------
        name : str, optional
            Filename to save the parameters, defaults to None.

        Returns:
        --------
        OrderedDict or None
            A dictionary containing the network parameters if `name` is not
            provided.
        """
        network_dict = OrderedDict()

        network_dict["ObjectTypeID"] = "Cerberus"
        network_dict["in_channels"] = self.in_channels
        network_dict["out_channels"] = self.out_channels
        network_dict["projection_layers"] = self.projection_layers
        network_dict["dropout"] = self.dropout
        network_dict["clip_low"] = self.clip_low
        network_dict["clip_high"] = self.clip_high
        network_dict["final_action"] = self.final_action
        network_dict["skip_connections"] = self.skip_connections
        network_dict[
            "base_network"] = self.base_network.save_network_parameters()
        network_dict["head_m1"] = self.head_m1.save_network_parameters()
        network_dict["head_0"] = self.head_0.save_network_parameters()
        network_dict["head_p1"] = self.head_p1.save_network_parameters()

        if self.partial_feature_projector is not None:
            network_dict["partial_feature_projector"] = (
                self.partial_feature_projector.save_parameters())

        if name is None:
            return network_dict
        torch.save(network_dict, name)

    def to(self, device):
        """
        Moves all model parameters, including the buffers from the
        PartialFeatureProjector, to the specified device.
        """
        # Call the parent class 'to' method
        super(Cerberus, self).to(device)

        # Ensure PartialFeatureProjector's buffers are moved as well
        if self.partial_feature_projector is not None:
            self.partial_feature_projector.mean = self.partial_feature_projector.mean.to(device)
            self.partial_feature_projector.sigma = self.partial_feature_projector.sigma.to(device)
            self.partial_feature_projector.projector = self.partial_feature_projector.projector.to(device)

        return self

def cerberus_sms_from_file(filename):
    """
    Load a Cerberus model from a saved file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the saved model.

    Returns:
    --------
    Cerberus
        An instance of the Cerberus model initialized from the saved
        parameters.
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))

    # Instantiate networks from saved parameters
    base_network = smsnet.SMSNetwork_from_file(network_dict["base_network"])
    lower = fcnet.FCNetwork_from_file(network_dict["head_m1"])
    median = fcnet.FCNetwork_from_file(network_dict["head_0"])
    upper = fcnet.FCNetwork_from_file(network_dict["head_p1"])

    # Check for partial feature projector
    partial_feature_projector = None
    if "partial_feature_projector" in network_dict:
        partial_feature_projector = PartialFeatureProjector_from_file(
            network_dict["partial_feature_projector"])

    # Instantiate the Cerberus class
    result = Cerberus(
        in_channels=network_dict["in_channels"],
        out_channels=network_dict["out_channels"],
        projection_layers=network_dict["projection_layers"],
        dropout=network_dict["dropout"],
        clip_low=network_dict["clip_low"],
        clip_high=network_dict["clip_high"],
        final_action=network_dict["final_action"],
        skip_connections=network_dict["skip_connections"],
        base_network=base_network,
        net_lower_quantile=lower,
        net_median=median,
        net_upper_quantile=upper,
        partial_feature_projector=partial_feature_projector
    )
    return result




def cerberus_unet_from_file(filename):
    """
    Load a Cerberus model from a saved file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the saved model.

    Returns:
    --------
    Cerberus
        An instance of the Cerberus model initialized from the saved
        parameters.
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))

    # Instantiate networks from saved parameters
    base_network = tunet.TUNetwork_from_file(network_dict["base_network"])
    lower = fcnet.FCNetwork_from_file(network_dict["head_m1"])
    median = fcnet.FCNetwork_from_file(network_dict["head_0"])
    upper = fcnet.FCNetwork_from_file(network_dict["head_p1"])

    # Check for partial feature projector
    partial_feature_projector = None
    if "partial_feature_projector" in network_dict:
        partial_feature_projector = PartialFeatureProjector_from_file(
            network_dict["partial_feature_projector"])

    # Instantiate the Cerberus class
    result = Cerberus(
        in_channels=network_dict["in_channels"],
        out_channels=network_dict["out_channels"],
        projection_layers=network_dict["projection_layers"],
        dropout=network_dict["dropout"],
        clip_low=network_dict["clip_low"],
        clip_high=network_dict["clip_high"],
        final_action=network_dict["final_action"],
        skip_connections=network_dict["skip_connections"],
        base_network=base_network,
        net_lower_quantile=lower,
        net_median=median,
        net_upper_quantile=upper,
        partial_feature_projector=partial_feature_projector
    )
    return result
