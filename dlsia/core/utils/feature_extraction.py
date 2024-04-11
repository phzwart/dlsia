import numpy as np
import napari
import einops
import torch
from qlty import qlty2D, cleanup
from skimage import feature
from functools import partial



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
                 sigma_min = 1,
                 sigma_max = 10,
                 num_sigma = 10,
                 normalize = True):
        super(feature_extractor, self).__init__()

        self.normalize = normalize
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_sigma = num_sigma

        self.extractor = partial(feature.multiscale_basic_features,
                            intensity=True,
                            edges=True,
                            texture=True,
                            sigma_min=self.sigma_min,
                            sigma_max=self.sigma_max,
                            num_sigma=self.num_sigma,
                            channel_axis=0)


    def _process_3d(self, images):
        assert len(images.shape)==5
        if torch.is_tensor(images) is False:
            images = torch.as_tensor(images)
        features = []
        for img in images:
            feature = torch.Tensor(self.extractor(img.numpy()))
            if self.normalize is True:
                means = torch.mean(feature, dim=(0, 1))
                stds = torch.std(feature, dim=(0, 1))
                feature = (feature - means) / stds
            features.append(feature.unsqueeze(0))
        features = torch.cat(features, dim=0)
        result = torch.concatenate( [images, einops.rearrange(features, "N Z Y X C -> N C Z Y X") ], dim=1 )

        return result

    def _process_2d(self, images):
        assert len(images.shape)==4
        if torch.is_tensor(images) is False:
            images = torch.as_tensor(images)

        features = []
        for img in images:
            feature = torch.Tensor(self.extractor(img.numpy()))
            if self.normalize is True:
                means = torch.mean(feature, dim=(0, 1))
                stds = torch.std(feature, dim=(0, 1))
                feature = (feature - means) / stds
                #feature = self.normal_cdf(feature)
            features.append(feature.unsqueeze(0))
        features = torch.cat(features, dim=0)
        m = torch.mean(images, dim=(0,-2,-1)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        s = torch.std(images, dim=(0,-2,-1)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        print(m.shape, s.shape, images.shape)
        images = (images-m)/s
        features = einops.rearrange(features, "N Y X C -> N C Y X")
        result = torch.concatenate( [images, features], dim=1 )
        return result

    def process(self, images):
        if len(images.shape)==4:
            return self._process_2d(images)
        if len(images.shape) == 5:
            return self._process_3d(images)

    def __call__(self, images):
        return self.process(images)


if __name__ == "__main__":
    img = torch.Tensor(np.random.random((2,1,100,100)))
    obj = feature_extractor(100,100,20)
    tmp = obj.process(img)



