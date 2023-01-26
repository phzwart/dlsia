import einops
import torch
from dlsia.core.networks import smsnet


class aggregate_model_preprocessor:
    """
    Build a preprocessor for AggNets
    """

    def __init__(self, cached_network_parameters):
        """
        Build a preprocessor for AggNets.

        :param cached_network_names: A list of pretrained SMS networks parameters are all trained to do a similar task.
        """
        self.cached_networks = []
        self.cached_network_parameters = cached_network_parameters
        for network_parameters in self.cached_network_parameters:
            with torch.no_grad():
                torch.cuda.empty_cache()
                tmp = smsnet.SMSNet(**network_parameters["topo_dict"])
                tmp.load_state_dict(network_parameters["state_dict"])
                torch.cuda.empty_cache()
                tmp.return_before_last_layer_ = True
                tmp.requires_grad_ = False
                self.cached_networks.append(tmp)

    def __call__(self, x, device):
        """
        Compute run each cached network on the supplied data, store, reshape and return results.
        Instead of using the actual results of the trained networks, we use the before-last layers.
        Data will always be returned as a tensor on the cpu.

        :param x: Input data tensors
        :param device: Where do we ruin the network?
        :return: output data (on the cpu by default)
        """
        result = []
        for net in self.cached_networks:
            torch.cuda.empty_cache()
            with torch.no_grad():
                net = net.to(device)
                tmp_layers, tmp_ps = net.to(device)(x.to(device))
                # tmp_ps = tmp_ps.cpu()
                tmp_layers = tmp_layers.cpu()
                # net = net.cpu()
                torch.cuda.empty_cache()
                for layer in tmp_layers.cpu()[0, ...]:
                    result.append(layer)
                del net
                del tmp_ps

        result = einops.rearrange(result, "C Y X -> C Y X")
        return result

    def save_network_parameters(self, name):
        """
        Save the preprocessor network parameters
        :param name: the filename
        :type name: str
        :return: None
        :rtype: None
        """
        torch.save(self.cached_network_parameters, name)


def aggregate_preprocessor_from_file(filename):
    """
    Build a aggregate preprocessor from file
    :param filename: the filename
    :type filename: str
    :return: an aggregate preprocessor
    :rtype: aggregate_model_preprocessor
    """
    params = torch.load(filename)
    obj = aggregate_model_preprocessor(params)
    return obj


def preprocess_data_aggregate_training(data_loader, agg_preprocessor, device):
    """
    Use an available data loader and a preconstructed preprocessors to build
    build new input and output data.

    :param data_loader: a data loader that returns x,y
    :param agg_preprocessor:
    :param device:
    :return:
    """
    x_all = []
    y_all = []
    y_shape = None
    for x, y in data_loader:
        if y_shape is None:
            y_shape = y.shape
        tmp_x = agg_preprocessor(x, device)
        x_all.append(tmp_x)
        y_all.append(y)
    x_all = einops.rearrange(x_all, "M C Y X -> M C Y X")

    if len(y_shape) == 4:
        y_all = einops.rearrange(y_all, "M N C Y X -> (M N) C Y X")
    if len(y_shape) == 3:
        y_all = einops.rearrange(y_all, "M N Y X -> (M N) Y X")

    return x_all, y_all
