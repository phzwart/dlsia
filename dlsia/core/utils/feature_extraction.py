import numpy as np
import napari
import einops
import torch
from qlty import qlty2D, cleanup
from skimage import feature
from functools import partial


def covariance_matrix(tensor):
    # Input tensor shape: (N, K, C)
    N, K, C = tensor.shape

    # Subtract the mean along the K dimension (centering the data)
    tensor_centered = tensor - tensor.mean(dim=1, keepdim=True)

    # Compute the covariance matrix for each sample in the batch
    cov_matrix = torch.matmul(tensor_centered.transpose(1, 2),
                              tensor_centered) / (K - 1)

    # Resulting shape: (N, C, C)
    return cov_matrix


class feature_extractor:
    """
    Computes a feature extraction on a stack of input images. Modules for viewing
    and extracting data are available

    :param image: stack of numpy arrays or torch tensors
    :param bool normalize: mean shift and normalize by standard deviation
    :param int window_size: size of qlty window for targets
    :param list sigmas: list of sigmas values in which to compute
    """

    def __init__(self,
                 image,
                 normalize=True,
                 window_size=50,
                 sigmas=[1, 2, 3, 4, 5]):
        self.image = image
        # self.extractor = extractor
        self.normalize = normalize
        self.window_size = window_size
        self.sigmas = sigmas

        super().__init__()

        if torch.is_tensor(image) is False:
            image = torch.as_tensor(image)

        print(len(image.size()))
        if len(image.size()) == 2:
            image = image.unsqueeze(0)

        # if extractor is None:
        sigma_min = self.sigmas[0]
        sigma_max = self.sigmas[-1]
        num_sigma = len(self.sigmas)
        extractor = partial(feature.multiscale_basic_features,
                            intensity=True,
                            edges=True,
                            texture=True,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            num_sigma=num_sigma,
                            channel_axis=0)

        features = torch.Tensor(extractor(image.numpy()))

        if normalize is True:
            means = torch.mean(features, dim=(0, 1))
            stds = torch.std(features, dim=(0, 1))
            features = (features - means) / stds

        qlt_obj = qlty2D.NCYXQuilt(Y=features.shape[0],
                                   X=features.shape[1],
                                   window=(window_size, window_size),
                                   step=(window_size, window_size),
                                   border=(1, 1),
                                   border_weight=0.5)

        t_X_all = einops.rearrange(features, "Y X N -> () N Y X")
        q_X_all = qlt_obj.unstitch(t_X_all)
        vecs = einops.rearrange(q_X_all, "N K Y X -> N (Y X) K")

        vcv = torch.logdet(covariance_matrix(vecs))

        variations = vcv
        v = variations.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        template = torch.ones((v.size()[0], 1, window_size, window_size)) * v
        output, _ = qlt_obj.stitch(template)

        self.output = output
        self.features = features
        self.t_X_all = t_X_all

    def view_napari(self):
        '''Views input, features, and target areas in Napari'''
        viewer = napari.view_image(self.image, name='Input')
        _ = viewer.add_image(self.output.numpy()[0, 0], name='Target',
                             colormap='inferno')
        _ = viewer.add_image(self.t_X_all.numpy(), name='Features')

    def get_target(self):
        return self.output

    def get_features(self):
        return self.t_X_all