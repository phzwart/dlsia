from dlsia.core.networks import fcnet
from dlsia.core.networks import smsnet
import torch
from torch import nn
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


def Squashnet_from_file_SMS(filename):
    network_dict = torch.load(filename, map_location=torch.device('cpu'))
    SMSObj = smsnet.SMSNet(**network_dict["topo_dict_spatial"])
    SMSObj.load_state_dict(network_dict["state_dict_spatial"])

    FCNet = fcnet.FCNetwork(**network_dict["topo_dict_squash"])
    FCNet.load_state_dict(network_dict["state_dict_squash"])

    result = SquashNet(FCNet, SMSObj)
    return result


