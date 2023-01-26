import matplotlib.pyplot as plt
import numpy as np
from dlsia.test_data.two_d import diffusion_model


class DataMaker(object):
    """
    A class that can be used to build training data for 2D peak picking
    classification of peaks that diffuse in a 2D plane. The returned data is a
    stack of 3D images, with two spatial coordinates and one time axis.
    Returned is noisy data, error free data and a 'mask'.
    """

    def __init__(self, n_peaks=1, sigma=0.05, trend=0.0, dxy=0.01, cc=0.5,
                 n_xy=128, bump=4.0):
        """
        Initializes a class that generates a stack of two_d images of randomly
        placed Gaussian shaped peaks.

        :param n_peaks: The number of peaks to be placed in the unit box.
        :param sigma: The width of the gaussian peak.
        :param trend: the length of a displacement trend vector
        :param dxy: diffusion parameter across time steps.
        :param cc: the correlation between displacements in x and y
        :param n_xy: The number of pixels in the unit box.
        :param bump: Controls border bumper to avoid placing peaks too close
                     to edge.
        """

        self.n_peaks = n_peaks
        self.n_xy = n_xy
        self.sigma = sigma

        self.trend = trend
        self.dxy = dxy
        self.cc = cc

        self.bump = bump
        self.xy = np.linspace(0, 1, n_xy)
        self.X, self.Y = np.meshgrid(self.xy, self.xy)

    def generate_ground_truth_image_stack(self, m_images, k_time_points, mask_radius=1.0):
        """
        Build a ground truth set of images, and associated binary mask of that
        image.

        :param m_images: Number of images to generate.
        :param k_time_points: Number of time points to generate
        :param mask_radius: used to build a binary mask derived from the ground
                            truth image.
        :return: ground truth images and associated mask.
        """

        # generate random positions
        positions = np.random.uniform(0 + self.bump * self.sigma,
                                      1 - self.bump * self.sigma, (self.n_peaks,
                                                                   2,
                                                                   m_images))

        # make arrays to store images
        ground_truth_images = np.zeros((m_images, k_time_points, self.n_xy,
                                        self.n_xy))
        ground_truth_mask = np.zeros((m_images, k_time_points, self.n_xy,
                                      self.n_xy))
        threshold_for_mask = np.exp(-mask_radius ** 2.0 / 2.0)

        # Lets walk over images, peak, time point

        for mm in range(m_images):
            # lets fix the displacement trend vector for all particles
            phi = np.random.uniform(0, np.pi * 2.0)
            t_x = self.trend * np.sin(phi)
            t_y = self.trend * np.cos(phi)

            for pp in range(self.n_peaks):
                this_x = positions[pp, 0, mm]
                this_y = positions[pp, 1, mm]
                displacements = diffusion_model.diffusion_2d(k_time_points,
                                                             mean_x=t_x,
                                                             mean_y=t_y,
                                                             sigma_x=self.dxy,
                                                             sigma_y=self.dxy,
                                                             cc=self.cc)
                for kk in range(k_time_points):
                    d_step = displacements[kk, :]
                    this_x = this_x + d_step[0]
                    this_y = this_y + d_step[1]

                    quadratic_form = (self.X - this_x) ** 2.0 + (self.Y - this_y) ** 2.0
                    quadratic_form = quadratic_form / (2.0 * self.sigma ** 2.0)
                    tmp_image = np.exp(-quadratic_form)

                    mask_sel = tmp_image >= threshold_for_mask
                    mask_image = np.zeros((self.n_xy, self.n_xy))
                    mask_image[mask_sel] = 1.0

                    ground_truth_images[mm, kk, :, :] += tmp_image
                    ground_truth_mask[mm, kk, :, :] += mask_image

        sel = ground_truth_mask > 0.5
        ground_truth_mask[sel] = 1.0
        return ground_truth_images, ground_truth_mask

    def generate_data_with_uniform_noise(self, m_images, k_time_steps,
                                         noise_level=1.0, mask_radius=1.0):
        """
        Build a dataset with uniform noise at a set level

        :param m_images: The number of images
        :param k_time_steps: The number of time steps
        :param noise_level: The noise level, data drawn from U(0,noise_level)
        :param mask_radius: The mask radius for ground_truth mask building
        :return: ground truth image, mask and noisy images
        """
        gt_img, gt_mask = self.generate_ground_truth_image_stack(m_images,
                                                                 k_time_steps,
                                                                 mask_radius)
        noise = np.random.uniform(0, noise_level, gt_img.shape)
        return gt_img, gt_mask, gt_img + noise


def tst(show=False, n_xy=64, n_times=10, n_img=1):
    """
    Provides a simple check if the peaks are generated
    Also serves as an indication how to use the above class.

    :param show: If True the full time series will be shown.
    :param n_xy: Spatial dimension of box.
    :param n_times: Number of time steps.
    :param n_img: Number of movies made.
    :return: True is test is passed.
    """
    n_peaks = 1
    sigma = 0.01

    dxy = 0.01
    trend = 0.02
    cc = 0.00

    img_engine = DataMaker(n_peaks,
                           sigma=sigma,
                           trend=trend,
                           dxy=dxy,
                           cc=cc,
                           n_xy=n_xy)
    imgs, msks, n_imgs = img_engine.generate_data_with_uniform_noise(n_img,
                                                                     n_times,
                                                                     noise_level=1,
                                                                     mask_radius=2.0)
    img_sum = np.sum(imgs, axis=(1, 2, 3))
    img_sum = np.mean(img_sum)
    theory = 2.0 * np.pi * sigma * sigma * n_peaks * n_xy * n_xy * n_times
    residual = abs(theory - img_sum) / theory

    imgs = imgs[0, :, :, :]
    n_imgs = n_imgs[0, :, :, :]
    msks = msks[0, :, :, :]

    if show:
        for img, msk, nimg in zip(imgs, msks, n_imgs):
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.imshow(img)
            plt.colorbar(shrink=0.70)
            plt.title('Ground Truth')

            plt.subplot(132)
            plt.imshow(msk)
            plt.colorbar(shrink=0.70)
            plt.title('Mask')

            plt.subplot(133)
            plt.imshow(nimg)
            plt.colorbar(shrink=0.70)
            plt.title('Noisy Image')

            plt.show()

    if not show:
        if residual > 1e-2:
            print(residual)
        assert (residual < 1e-2)
        return True

    return None


def run_tst(n_xy=1000, n_times=10, n_imgs=1):
    tst(False, n_xy, n_times, n_imgs)


if __name__ == "__main__":
    run_tst()
    tst(True)
