import numpy as np
import dlsia
from dlsia.core.networks import sms1d
from dlsia.core.networks import smsnet
import torch
from torch import nn
import einops
from collections import OrderedDict

class SquashNet(nn.Module):
    def __init__(self,
                 squash_head,
                 spatial_net,
                 ):
        super(SquashNet, self).__init__()
        self.squash_head = squash_head
        self.spatial_net = spatial_net

    def forward(self,x):
        x = self.squash_head(x)
        x = self.spatial_net(x)
        return x


    def save_network_parameters(self, name=None):
        """
        Save the network parameters
        :param name: The filename
        :type name: str
        :return: None
        :rtype: None
        """
        network_dict = OrderedDict()
        network_dict["topo_dict_squash"] = self.squash_head.topology_dict()
        network_dict["state_dict_squash"] = self.squash_head.state_dict()
        network_dict["topo_dict_spatial"] = self.spatial_net.topology_dict()
        network_dict["state_dict_spatial"] = self.spatial_net.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


class FCNetwork(nn.Module):
    def __init__(self, Cin, Cmiddle, Cout, dropout_rate=0.5):
        super(FCNetwork, self).__init__()

        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = Cout
        self.dropout_rate = dropout_rate

        layers = []
        layers.append(nn.Linear(self.Cin, self.Cmiddle[0]))
        layers.append(nn.BatchNorm1d(self.Cmiddle[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        for i in range(len(self.Cmiddle)-1):
            layers.append(nn.Linear(self.Cmiddle[i], self.Cmiddle[i+1]))
            layers.append(nn.BatchNorm1d(self.Cmiddle[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(self.Cmiddle[-1], self.Cout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        N,C,Y,X = x.shape
        x = einops.rearrange(x, "N C Y X -> (N Y X) C")
        x = self.network(x)
        x = einops.rearrange(x, "(N Y X) C -> N C Y X", N=N, Y=Y, X=X)
        return x

    def topology_dict(self):
        topo = OrderedDict()
        topo["Cin"] = self.Cin
        topo["Cmiddle"] = self.Cmiddle
        topo["Cout"] = self.Cout
        topo["dropout_rate"] = self.dropout_rate
        return topo

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)

def Squashnet_from_file_SMS(filename):
    network_dict = torch.load(filename, map_location=torch.device('cpu'))
    SMSObj = smsnet.SMSNet(**network_dict["topo_dict_spatial"])
    SMSObj.load_state_dict(network_dict["state_dict_spatial"])

    FCNet = FCNetwork(**network_dict["topo_dict_squash"])
    FCNet.load_state_dict(network_dict["state_dict_squash"])

    result = SquashNet(FCNet, SMSObj)
    return result


