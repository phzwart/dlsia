"""
Tools for sparse mixed scale networks where size changes of tensors are needed
"""
import einops
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from dlsia.core.networks import smsnet


class SparseLabeler(nn.Module):
    """
    This object takes a (C Y X) tensor, compresses it to a vector of
    length class_size
    """

    def __init__(self,
                 in_shape=(28, 28),
                 latent_shape=(7, 7),
                 in_channels=1,
                 out_classes=10,
                 depth=20,
                 dilations=None,
                 hidden_channels=10,
                 out_channels=1,  #
                 alpha_range=(0.75, 1.0),
                 gamma_range=(0.0, 0.5),
                 max_degree=5,
                 min_degree=1,
                 pIL=0.25,
                 pLO=0.25,
                 IO=True,
                 stride_base=2,
                 dropout_rate=0.05,
                 encoder=None
                 ):
        """
        Take an tensor and assign class probabilities for object identification.
        Use the randomized Sparse Mixed Scale network as backbone

        Parameters
        ----------
        in_shape : input shape
        latent_shape : output shape before its turned into a linear vector
        out_classes : number of classes
        depth : the depth of the network
        dilations : choices of dilations
        hidden_channels : hidden channels
        out_channels : number of output channels
        alpha_range : range in which alpha lies (controls length of skip
                      connections)
        gamma_range : range in whgich gamma lies (controls distribution of
                      node degree)
        max_degree : maximum degree per node
        min_degree : minimum degree per node
        pIL : probability of connection between start of network and hidden
              node
        pLO : probability of connection between hidden node and output node
        IO : bool that controls connection between input tensor and output
             tensor
        stride_base : The base of possible strides.
        dropout_rate : dropout rate for FC layers in the end.
        """
        super(SparseLabeler, self).__init__()

        self.in_shape = in_shape
        self.latent_shape = latent_shape
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.dropout_rate = dropout_rate
        self.out_channels = out_channels

        if encoder is None:
            if dilations is None:
                dilations = [1, 2, 3]

            size_ratio1 = in_shape[0] / latent_shape[0]
            size_ratio2 = in_shape[1] / latent_shape[1]
            assert np.abs(size_ratio1 - size_ratio2) < 1e-3

            power = np.floor(np.log(size_ratio1) / np.log(stride_base) + 0.5)
            min_power = -int(power)

            h_channels = [hidden_channels]
            layer_probabilities = {'LL_alpha': np.random.uniform(alpha_range[0],
                                                                 alpha_range[1]),
                                   'LL_gamma': np.random.uniform(gamma_range[0],
                                                                 gamma_range[1]),
                                   'LL_max_degree': max_degree,
                                   'LL_min_degree': min_degree,
                                   'IL': pIL,
                                   'LO': pLO,
                                   'IO': IO}

            sizing_settings1 = {'stride_base': stride_base,
                                'min_power': min_power,
                                'max_power': 0}

            self.encode = smsnet.random_SMS_network(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    in_shape=in_shape,
                                                    out_shape=latent_shape,
                                                    sizing_settings=sizing_settings1,
                                                    layers=depth,
                                                    dilation_choices=dilations,
                                                    layer_probabilities=layer_probabilities,
                                                    hidden_out_channels=h_channels,
                                                    network_type="Regression")
        else:
            self.encode = encoder

        self.Nlatent = out_channels * latent_shape[0] * latent_shape[1]
        self.do1 = nn.Dropout1d(p=self.dropout_rate)
        self.bn1 = nn.BatchNorm1d(num_features=self.Nlatent)
        self.fc1 = nn.Linear(self.Nlatent, self.Nlatent)
        self.do2 = nn.Dropout1d(p=self.dropout_rate)
        self.bn2 = nn.BatchNorm1d(num_features=self.out_classes)
        self.fc2 = nn.Linear(self.Nlatent, self.out_classes)

    def forward(self, x):
        """
        Standard forward operator

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        Class probabilities (non-softmax)

        """
        xae = self.encode(x)
        pclass = self.do1(einops.rearrange(xae, "N C Y X -> N (C Y X)"))
        pclass = self.do2(nn.ReLU()(self.bn1(self.fc1(pclass))))
        pclass = self.bn2(self.fc2(pclass))
        return pclass

    def to(self, device):
        """
        A to function.

        Parameters
        ----------
        device : target device

        Returns
        -------
        object on device
        """

        self.encode = self.encode.to(device)
        self.bn1 = self.bn1.to(device)
        self.bn2 = self.bn2.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        return self

    def topology_dict(self):
        topo_dict = OrderedDict()
        topo_dict["in_shape"] = self.in_shape
        topo_dict["latent_shape"] = self.latent_shape
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_classes"] = self.out_classes
        topo_dict["dropout_rate"] = self.dropout_rate
        topo_dict["out_channels"] = self.out_channels
        return topo_dict

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["encoder"] = self.encode.topology_dict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


def SparseLabeler_from_file(filename):
    """
    Construct an SMSNet from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An SMSNet
    :rtype: smsnet
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))

    encoder = smsnet.SMSNet(**network_dict["encoder"])
    SPL = SparseLabeler(encoder=encoder, **network_dict["topo_dict"])
    SPL.load_state_dict(network_dict["state_dict"])
    return SPL


def fc_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class SparseAEC(nn.Module):
    """
    A randomized Sparse Mixed Scale Autoencoder, with a classifier attached to
    the compressed image latent space
    """

    def __init__(self,
                 in_shape=(28, 28),
                 latent_shape=(7, 7),
                 in_channels=1,
                 out_classes=10,
                 depth=20,
                 dilations=None,
                 hidden_channels=10,
                 out_channels=1,
                 alpha_range=(0.75, 1.0),
                 gamma_range=(0.0, 0.5),
                 max_degree=5,
                 min_degree=1,
                 pIL=1.0,
                 pLO=1.0,
                 IO=True,
                 stride_base=2,
                 dropout_rate=0.05,
                 encoder=None,
                 decoder=None
                 ):
        """
        A randomized Sparse Mixed Scale Autoencoder, with a classifier attached
        to the latent space of the compressed image.

        Parameters
        ----------
        in_shape : input shape
        latent_shape : output shape of the encoder
        out_classes : number of classes
        depth : the depth of the network
        dilations : choices of dilations
        hidden_channels : hidden channels
        out_channels : number of output channels
        alpha_range : range in which alpha lies (controls length of skip
                      connections)
        gamma_range : range in whgich gamma lies (controls distribution of
                      node degree)
        max_degree : maximum degree per node
        min_degree : minimum degree per node
        pIL : probability of connection between start of network and hidden
              node
        pLO : probability of connection between hidden node and output node
        IO : bool that controls connection between input tensor and output
             tensor
        stride_base : The base of possible strides.
        dropout_rate: The dropout rate for dropout layers in the classification part of the network.
        """

        super(SparseAEC, self).__init__()
        self.in_shape = in_shape
        self.latent_shape = latent_shape
        self.out_classes = out_classes
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.stride_base = stride_base

        if dilations is None:
            dilations = [1, 2, 3]
        size_ratio1 = in_shape[0] / latent_shape[0]
        size_ratio2 = in_shape[1] / latent_shape[1]
        assert np.abs(size_ratio1 - size_ratio2) < 1e-3

        power = np.floor(np.log(size_ratio1) / np.log(stride_base) + 0.5)
        min_power = -int(power)
        max_power = int(power)

        if encoder is None:
            assert decoder is None
            h_channels = [hidden_channels]
            layer_probabilities = {'LL_alpha': np.random.uniform(alpha_range[0],
                                                                 alpha_range[1]),
                                   'LL_gamma': np.random.uniform(gamma_range[0],
                                                                 gamma_range[1]),
                                   'LL_max_degree': max_degree,
                                   'LL_min_degree': min_degree,
                                   'IL': pIL,
                                   'LO': pLO,
                                   'IO': IO}

            sizing_settings1 = {'stride_base': stride_base,
                                'min_power': min_power,
                                'max_power': 0}

            sizing_settings2 = {'stride_base': stride_base,
                                'min_power': 0,
                                'max_power': max_power}

            self.encode = smsnet.random_SMS_network(in_channels=self.in_channels,
                                                    out_channels=out_channels,
                                                    in_shape=in_shape,
                                                    out_shape=latent_shape,
                                                    sizing_settings=sizing_settings1,
                                                    layers=depth,
                                                    dilation_choices=dilations,
                                                    layer_probabilities=layer_probabilities,
                                                    hidden_out_channels=h_channels,
                                                    network_type="Regression")

            self.decode = smsnet.random_SMS_network(in_channels=out_channels,
                                                    out_channels=self.in_channels,
                                                    in_shape=latent_shape,
                                                    out_shape=in_shape,
                                                    sizing_settings=sizing_settings2,
                                                    layers=depth,
                                                    dilation_choices=dilations,
                                                    layer_probabilities=layer_probabilities,
                                                    hidden_out_channels=h_channels,
                                                    network_type="Regression")
        else:
            self.encode = encoder
            self.decode = decoder

        self.Nlatent = out_channels * latent_shape[0] * latent_shape[1]

        self.bn1 = nn.BatchNorm1d(num_features=self.Nlatent // 2)
        self.fc1 = nn.Linear(self.Nlatent, self.Nlatent // 2)
        self.dropout1 = nn.Dropout1d(self.dropout_rate)

        self.bn2 = nn.BatchNorm1d(num_features=self.Nlatent // 2)
        self.fc2 = nn.Linear(self.Nlatent // 2, self.Nlatent // 2)
        self.dropout2 = nn.Dropout1d(self.dropout_rate)

        self.fc3 = nn.Linear(self.Nlatent // 2, out_classes)
        self.dropout3 = nn.Dropout1d(self.dropout_rate)

    def reinitialize_fc(self):
        self.apply(fc_weights_init)

    def forward(self, x):
        """
        Standard forward operator

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        Class probabilities (non-softmax)

        """
        xae = self.encode(x)
        pclass = einops.rearrange(xae, "N C Y X -> N (C Y X)")
        pclass = self.dropout2(nn.ReLU()(self.bn1(self.fc1(pclass))))
        pclass = self.dropout3(nn.ReLU()(self.bn2(self.fc2(pclass))))
        pclass = self.fc3(pclass)
        xae = self.decode(xae)
        return xae, pclass

    def latent(self, x):
        xae = self.encode(x)
        xae = einops.rearrange(xae, "N C Y X -> N (C Y X)")
        return xae

    def to(self, device):
        """
        A to function

        Parameters
        ----------
        device : target device

        Returns
        -------
        object on device
        """

        self.encode = self.encode.to(device)
        self.decode = self.decode.to(device)
        self.bn1 = self.bn1.to(device)
        self.bn2 = self.bn2.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.fc3 = self.fc3.to(device)

        self.dropout1 = self.dropout1.to(device)
        self.dropout2 = self.dropout2.to(device)
        self.dropout3 = self.dropout3.to(device)

        return self

    def topology_dict(self):
        topo_dict = OrderedDict()

        topo_dict["in_shape"] = self.in_shape
        topo_dict["latent_shape"] = self.latent_shape
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_classes"] = self.out_classes
        topo_dict["dropout_rate"] = self.dropout_rate
        topo_dict["out_channels"] = self.out_channels
        topo_dict["stride_base"] = self.stride_base
        return topo_dict

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["encode"] = self.encode.save_network_parameters()
        network_dict["decode"] = self.decode.save_network_parameters()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


def SparseAEC_from_file(filename):
    """
    Construct an SparseAEC from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An SparseAEC
    :rtype: SparseAEC
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))

    encoder = smsnet.SMSNet(**network_dict["encode"]["topo_dict"])
    decoder = smsnet.SMSNet(**network_dict["decode"]["topo_dict"])

    SAEC = SparseAEC(encoder=encoder, decoder=decoder, **network_dict["topo_dict"])
    SAEC.load_state_dict(network_dict["state_dict"])
    return SAEC


class SparseAutoEncoder(nn.Module):
    """
    An Autoencoder based on randomized Sparse Mixed Scale Networks
    """

    def __init__(self,
                 in_shape=(28, 28),
                 latent_shape=(7, 7),
                 latent_sequence = [32,2],
                 dropout_p=0.05,
                 in_channels=1,
                 depth=20,
                 dilations=None,
                 hidden_channels=10,
                 out_channels=1,
                 alpha_range=(0.75, 1.0),
                 gamma_range=(0.0, 0.5),
                 max_degree=5,
                 min_degree=1,
                 pIL=1.0,
                 pLO=1.0,
                 IO=True,
                 stride_base=2,
                 final_transform=None,
                 encoder=None,
                 decoder=None
                 ):
        """
        A randomized Sparse Mixed Scale Autoencoder, with a classifier attached
        to the latent space of the compressed image.

        Parameters
        ----------
        in_shape : input shape
        latent_shape : output shape of the encoder
        depth : the depth of the network
        dilations : choices of dilations
        hidden_channels : hidden channels
        out_channels : number of output channels
        alpha_range : range in which alpha lies (controls length of skip
                      connections)
        gamma_range : range in which gamma lies (controls distribution of
                      node degree)
        max_degree : maximum degree per node
        min_degree : minimum degree per node
        pIL : probability of connection between start of network and hidden
              node
        pLO : probability of connection between hidden node and output node
        IO : bool that controls connection between input tensor and output
             tensor
        final_transform : An additional transform on the output object if
                          desired, like Sigmoid, ReLU etc
        """

        super(SparseAutoEncoder, self).__init__()
        self.in_shape = in_shape
        self.latent_shape = latent_shape
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride_base = stride_base
        self.final_transform = final_transform
        self.latent_sequence = latent_sequence
        self.dropout_p = dropout_p

        if encoder is None:
            assert decoder is None

            if dilations is None:
                dilations = [1, 2, 3]

            size_ratio1 = in_shape[0] / latent_shape[0]
            size_ratio2 = in_shape[1] / latent_shape[1]
            assert np.abs(size_ratio1 - size_ratio2) < 1e-3

            power = np.floor(np.log(size_ratio1) / np.log(stride_base) + 0.5)
            min_power = -int(power)
            max_power = int(power)

            h_channels = [hidden_channels]
            layer_probabilities = {'LL_alpha': np.random.uniform(alpha_range[0],
                                                                 alpha_range[1]),
                                   'LL_gamma': np.random.uniform(gamma_range[0],
                                                                 gamma_range[1]),
                                   'LL_max_degree': max_degree,
                                   'LL_min_degree': min_degree,
                                   'IL': pIL,
                                   'LO': pLO,
                                   'IO': IO}

            sizing_settings1 = {'stride_base': stride_base,
                                'min_power': min_power,
                                'max_power': 0}

            sizing_settings2 = {'stride_base': stride_base,
                                'min_power': 0,
                                'max_power': max_power}

            self.encode = smsnet.random_SMS_network(in_channels=self.in_channels,
                                                    out_channels=self.out_channels,
                                                    in_shape=self.in_shape,
                                                    out_shape=self.latent_shape,
                                                    sizing_settings=sizing_settings1,
                                                    layers=depth,
                                                    dilation_choices=dilations,
                                                    layer_probabilities=layer_probabilities,
                                                    hidden_out_channels=h_channels,
                                                    network_type="Regression")

            self.decode = smsnet.random_SMS_network(in_channels=self.out_channels,
                                                    out_channels=self.in_channels,
                                                    in_shape=self.latent_shape,
                                                    out_shape=self.in_shape,
                                                    sizing_settings=sizing_settings2,
                                                    layers=depth,
                                                    dilation_choices=dilations,
                                                    layer_probabilities=layer_probabilities,
                                                    hidden_out_channels=h_channels,
                                                    network_type="Regression")
        else:
            self.encode = encoder
            self.decode = decoder

        self.Nlatent = out_channels * latent_shape[0] * latent_shape[1]

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (self.out_channels, *self.latent_shape))


        layers = []
        previous_layer_features = self.Nlatent
        for num_features in latent_sequence[:-1]:
            layers.append(nn.Linear(previous_layer_features, num_features))
            layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            previous_layer_features = num_features
        layers.append(nn.Linear(previous_layer_features, latent_sequence[-1]))
        self.encoder_fc_layers = nn.Sequential(*layers)

        # Additional Fully Connected Layers for Decoder
        layers = []
        previous_layer_features = latent_sequence[-1]
        for num_features in reversed(latent_sequence[:-1]):
            layers.append(nn.Linear(previous_layer_features, num_features))
            layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            previous_layer_features = num_features
        layers.append(nn.Linear(previous_layer_features, self.Nlatent))
        self.decoder_fc_layers = nn.Sequential(*layers)

    def latent_vector(self, x):
        """
        Get the latent space vector

        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output latent space vector
        """
        x = self.encode(x)
        x = einops.rearrange(x, "N C Y X -> N (C Y X)")
        x = self.encoder_fc_layers(x)
        return x

    def forward(self, x):
        """
        A standard forward operator

        Parameters
        ----------
        x : input tensor

        Returns
        -------

        """

        x = self.encode(x)
        x = self.flatten(x)
        x = self.encoder_fc_layers(x)
        x = self.decoder_fc_layers(x)
        x = self.unflatten(x)
        x = self.decode(x)
        if self.final_transform is not None:
            x = self.final_transform(x)
        return x

    def to(self, device):
        """
        A To function
        Parameters
        ----------
        device : target device

        Returns
        -------
        object on device
        """
        self.encode = self.encode.to(device)
        self.decode = self.decode.to(device)
        self.encoder_fc_layers = self.encoder_fc_layers.to(device)
        self.decoder_fc_layers = self.decoder_fc_layers.to(device)
        return self

    def topology_dict(self):
        topo_dict = OrderedDict()
        topo_dict["in_shape"] = self.in_shape
        topo_dict["latent_shape"] = self.latent_shape
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_channels"] = self.out_channels
        topo_dict["stride_base"] = self.stride_base
        topo_dict["final_transform"] = self.final_transform

        return topo_dict

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["encode"] = self.encode.save_network_parameters()
        network_dict["decode"] = self.decode.save_network_parameters()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


def SparseAutoEncoder_from_file(filename):
    """
    Construct an SparseAEC from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An SparseAEC
    :rtype: SparseAEC
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))

    encoder = smsnet.SMSNet(**network_dict["encode"]["topo_dict"])
    decoder = smsnet.SMSNet(**network_dict["decode"]["topo_dict"])

    SAE = SparseAutoEncoder(encoder=encoder, decoder=decoder, **network_dict["topo_dict"])
    SAE.load_state_dict(network_dict["state_dict"])
    return SAE

