import torch
from torch import nn
import einops
from collections import OrderedDict


class FCNetwork(nn.Module):
    def __init__(self,
                 Cin,
                 Cmiddle,
                 Cout,
                 dropout_rate=0.0,
                 skip_connections=True,
                 o_channels=0
                 ):

        super(FCNetwork, self).__init__()

        self.Cin = Cin
        self.Cmiddle = Cmiddle
        self.Cout = Cout
        self.dropout_rate = dropout_rate
        self.skip_connections = skip_connections
        self.o_channels = o_channels

        linear_layer = nn.Linear # needed for monotonic networks. will use later keep it eventhough ugly.

        layers = []

        layers.append(linear_layer(self.Cin+self.o_channels, self.Cmiddle[0]))
        layers.append(nn.BatchNorm1d(self.Cmiddle[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        for i in range(len(self.Cmiddle)-1):
            layers.append(linear_layer(self.Cmiddle[i]+self.o_channels, self.Cmiddle[i+1]))
            layers.append(nn.BatchNorm1d(self.Cmiddle[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        layers.append(linear_layer(self.Cmiddle[-1]+self.o_channels, self.Cout))
        self.network = nn.Sequential(*layers)

    def forward1d(self, x, o=None):
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if self.skip_connections:
                    x = torch.cat((x, o), dim=-1)
            x = layer(x)
        return x

    def forward2d(self, x, o=None):
        Nx,Cx,Yx,Xx = x.shape
        x = einops.rearrange(x, "N C Y X -> (N Y X) C")
        if o is not None:
            No, Co, Yo, Xo = o.shape
            assert No == Nx
            assert Yo == Yx
            assert Xo == Xx
            o = einops.rearrange(o, "N C Y X -> (N Y X) C")

        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if self.skip_connections:
                    x = torch.cat((x, o), dim=-1)
            x = layer(x)
        x = einops.rearrange(x, "(N Y X) C -> N C Y X", N=Nx, Y=Yx, X=Xx)
        return x

    def forward3d(self, x, o=None):
        Nx,Cx,Zx,Yx,Xx = x.shape
        x = einops.rearrange(x, "N C Z Y X -> (N Z Y X) C")
        if o is not None:
            No, Co, Zo, Yo, Xo = o.shape
            assert No == Nx
            assert Yo == Yx
            assert Xo == Xx
            o = einops.rearrange(o, "N C Z Y X -> (N Z Y X) C")

        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                if self.skip_connections:
                    x = torch.cat((x, o), dim=-1)
            x = layer(x)
        x = einops.rearrange(x, "(N Z Y X) C -> N C Z Y X", N=Nx, Z=Zx, Y=Yx, X=Xx)
        return x


    def forward(self, x, o=None):
        if len(x.shape) == 4:
            return self.forward2d(x, o)
        if len(x.shape) == 5:
            return self.forward3d(x, o)
        if len(x.shape) == 2:
            return self.forward1d(x, o)

    def topology_dict(self):
        topo = OrderedDict()
        topo["Cin"] = self.Cin
        topo["Cmiddle"] = self.Cmiddle
        topo["Cout"] = self.Cout
        topo["dropout_rate"] = self.dropout_rate
        topo["skip_connections"] = self.skip_connections
        topo["o_channels"] = self.o_channels

        return topo

    def save_network_parameters(self, name=None):
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)

def FCNetwork_from_file(filename):
    if isinstance(filename, OrderedDict):
        network_dict = filename
    else:
        network_dict = torch.load(filename, map_location=torch.device('cpu'))
    result = FCNetwork(**network_dict["topo_dict"])
    result.load_state_dict(network_dict["state_dict"])
    return result