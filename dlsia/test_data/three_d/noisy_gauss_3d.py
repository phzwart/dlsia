import matplotlib.pyplot as plt
import numpy as np


class DataMaker3D(object):
    """
    A class that can be used to build training data for 2D peak picking
    classification. The returned data is a stack of 2D images consisting of
    noisy data, error free data and a 'mask'.
    """

    def __init__(self, n_peaks=1, sigma_xy=1.26, sigma_z=1.34, n_xyz=32,
                 bump=4.0):
        """
        Initializes a class that generates a stack of two_d images of randomly
        placed Gaussian shaped peaks.

        :param n_peaks: The number of peaks to be placed in the unit box.
        :type n_peaks: int
        :param sigma_xy: The width of the gaussian peak.
        :type sigma_xy: float
        :param sigma_z: The width of the gaussian peak.
        :type sigma_z: float
        :param n_xyz: The number of pixels in the unit cube.
        :param bump: Controls border bumper to avoid placing peaks too close
                     to the border.
        """

        self.Npeaks = n_peaks
        self.n_xyz = n_xyz
        self.sigma_xy = sigma_xy / n_xyz
        self.sigma_z = sigma_z / n_xyz
        self.bump = bump
        self.xyz = np.linspace(0, 1, self.n_xyz)
        self.X, self.Y, self.Z = np.meshgrid(self.xyz, self.xyz, self.xyz)

    def generate_ground_truth_image_stack(self, m_images, mask_radius=1.0):
        """
        Build a ground truth set of images, and associated binary mask

        :param m_images: Number of images to generate.
        :param mask_radius: used to build a binary mask derived from the ground
                            truth image.
        :return: ground truth images and associated mask.
        """

        # generate random positions
        positions = np.random.uniform(0 + self.bump * self.sigma_z,
                                      1 - self.bump * self.sigma_z, (self.Npeaks, 3, m_images))

        # make arrays to store images
        ground_truth_images = np.zeros((m_images, self.n_xyz, self.n_xyz, self.n_xyz))
        ground_truth_mask = np.zeros((m_images, self.n_xyz, self.n_xyz, self.n_xyz))
        threshold_for_mask = np.exp(-mask_radius ** 2.0 / 2.0)

        wxy = 1.0 / (self.sigma_xy ** 2.0)
        wz = 1.0 / (self.sigma_z ** 2.0)

        for peak in range(self.Npeaks):
            mux = positions[peak, 0, :]
            muy = positions[peak, 1, :]
            muz = positions[peak, 2, :]

            for mm in range(m_images):
                quadratic_form = wxy * (self.X - mux[mm]) ** 2.0 + wxy * (self.Y - muy[mm]) ** 2.0 + wz * (
                            self.Z - muz[mm]) ** 2.0
                quadratic_form = quadratic_form / 2.0
                tmp_image = np.exp(-quadratic_form)
                mask_sel = tmp_image >= threshold_for_mask
                mask_image = np.zeros((self.n_xyz, self.n_xyz, self.n_xyz))
                mask_image[mask_sel] = 1.0
                ground_truth_images[mm, :, :, :] += tmp_image
                ground_truth_mask[mm, :, :, :] += mask_image
        sel = ground_truth_mask > 0.5
        ground_truth_mask[sel] = 1.0

        return ground_truth_images, ground_truth_mask

    def generate_data_with_gaussian_noise(self,
                                          m_images,
                                          snr=1.0,
                                          mask_radius=2.0,
                                          noise_base=100.0,
                                          noise_sigma=4.0):
        """
        Build a dataset with uniform noise at a set level

        :param m_images: the number of images
        :param snr: the noise level, data drawn from U(0,noise_level)
        :param mask_radius: the mask radius for ground_truth mask building
        :param noise_base: mean of Gaussian distribution
        :param noise_sigma: standard deviation of the Gaussian distribution
        :return: ground truth image, mask and noisy images
        """

        gt_img, gt_mask = self.generate_ground_truth_image_stack(m_images,
                                                                 mask_radius)
        peak_value = snr * noise_sigma
        gt_img = gt_img * peak_value
        noise = np.random.normal(noise_base, noise_sigma, gt_img.shape)
        return gt_img, gt_mask, gt_img + noise


def tst(show=True, n_xyz=32, n_imgs=10):
    """
    Provides a simple check if the peaks are generated
    Also serves as an indication how to use the above class.

    :param show: If True, 3 sample images will be shown.
    :param n_xyz: Dimensions of the box
    :param n_imgs: Number of images generated
    :return: True is test is passed
    """
    n_peaks = 3

    img_engine = DataMaker3D(n_peaks=n_peaks, n_xyz=n_xyz)
    imgs, msks, n_imgs = img_engine.generate_data_with_gaussian_noise(n_imgs,
                                                                      snr=1.0)

    if show:
        for ii in range(n_xyz):
            img = imgs[0, ii, :, :]
            nimg = n_imgs[0, ii, :, :]
            msk = msks[0, ii, :, :]

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


def run_tst(n_xy=1000, n_imgs=1):
    tst(False, n_xy, n_imgs)


if __name__ == "__main__":
    # run_tst()
    tst(True, 32, 1)
