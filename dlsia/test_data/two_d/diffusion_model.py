import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def diffusion_2d(time_steps: int,
                 mean_x: float = 0.0,
                 mean_y: Union[None, float] = None,
                 sigma_x: float = 0.01,
                 sigma_y: float = 0.01,
                 cc: float = 0.0) -> np.ndarray:
    """
    Generates a sequence of displacement steps.
    If mean_y = None, a random vector will be generated with length mean_x

    :param time_steps: The number of displacements to be generated
    :param mean_x: The mean displacement in x
    :param mean_y: The mean displacement in y. If set to None, a random vector
                   will be generated.
    :param sigma_x: The sigma in x
    :param sigma_y: The sigma in y
    :param cc: The correlation between the two
    :return: an array with displacements
    """

    s = np.array([[sigma_x * sigma_x, cc * sigma_x * sigma_y],
                  [cc * sigma_x * sigma_y, sigma_y * sigma_y]])
    mu = np.array([0, 0])
    if mean_y is None:
        assert mean_x is not None
        phi = np.random.uniform(0, 2.0 * np.pi, 1)[0]
        x = np.sin(phi) * mean_x
        y = np.cos(phi) * mean_x
        mu = np.array([x, y]).flatten()
    if mean_y is not None:
        mu = np.array([mean_x, mean_y]).flatten()
    dxy = np.random.multivariate_normal(mu, s, time_steps)
    return dxy


def tst(show=False):
    kk = 100000
    dxy = diffusion_2d(time_steps=kk,
                       mean_x=1.0,
                       mean_y=None,
                       sigma_x=1.0,
                       sigma_y=1.00,
                       cc=0.7)
    mean = np.mean(dxy, axis=0)
    length = np.sqrt(np.sum(mean * mean))

    if show:
        plt.hist2d(dxy[:, 0], dxy[:, 1], bins=100)
        plt.show()

        for ii in range(1, kk):
            dxy[ii, 0] = dxy[ii - 1, 0] + dxy[ii, 0]
            dxy[ii, 1] = dxy[ii - 1, 1] + dxy[ii, 1]
        plt.plot(dxy[:, 0], dxy[:, 1], '.-')
        plt.show()
        return None
    else:
        assert abs(length - 1.0) < 1e-2
        return True


if __name__ == "__main__":
    tst(False)
