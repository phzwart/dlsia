import torch
from torch import nn
import einops
from collections import OrderedDict

class FCNetwork(nn.Module):
    def __init__(self, Cin, Cmiddle, Cout, dropout_rate=0.0):
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
