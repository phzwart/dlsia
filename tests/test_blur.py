import numpy as np
import torch
from dlsia.core.utils import blur
from scipy.ndimage import gaussian_filter

import pytest

def test_3d():
    N = 64
    img = torch.zeros((1,N,N,N))
    img[:, 20:-20,20:-20,20:-20] = 1.0
    mask = img[0].numpy()
    windows = np.arange(1,15)*2+1
    sigmas = np.linspace(0.25, 5.25, 10)
    eps = 1e-12
    threshold = 6

    for sigma in sigmas:
        for window in windows:
            tmp = blur.GaussianBlur3D(sigma, window)(img)
            tmp2 = gaussian_filter(img[0].numpy(), sigma)
            d = np.abs( tmp.detach().numpy() - tmp2 ) / (tmp2+eps)
            d = d*mask
            if window / sigma >  threshold:
                assert np.max(d) < 1.0e-2

def test_2d():
    N = 64
    img = torch.zeros((1,N,N))
    img[:, 20:-20,20:-20] = 1.0
    mask = img[0].numpy()
    windows = np.arange(1,15)*2+1
    sigmas = np.linspace(2.25, 5.25, 8)
    eps = 1e-12
    threshold = 6

    for sigma in sigmas:
        for window in windows:
            tmp = blur.GaussianBlur2D(sigma, window)(img)
            tmp2 = gaussian_filter(img[0].numpy(), sigma)
            d = np.abs( tmp.detach().numpy() - tmp2 ) / (tmp2+eps)
            d = d*mask


            if window / sigma >  threshold:
                assert np.max(d) < 1.0e-2


if __name__ == "__main__":
    test_2d()
