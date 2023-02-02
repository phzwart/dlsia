import matplotlib.pyplot as plt
import numpy as np


class DataMaker(object):
    """
    A class that can be used to build training data for 2D peak picking
    classification. The returned data is a stack of 2D images consisting of
    noisy data, error free data and a 'mask'.
    """

    def __init__(self, n_peaks=1, sigma=1.26, n_xy=128, bump=0.0):
        """
        Initializes a class that generates a stack of two_d images of randomly
        placed Gaussian shaped peaks.

        :param n_peaks: The number of peaks to be placed in the unit box.
        :param sigma: The width of the gaussian peak.
        :param n_xy: The number of pixels in the unit box.
        :param bump: Controls border bumper to avoid placing peaks too close to
                     the edge.
        """

        self.n_peaks = n_peaks
        self.n_xy = n_xy
        self.sigma = sigma
        self.bump = bump
        self.xy = np.linspace(0, n_xy - 1, n_xy)
        self.X, self.Y = np.meshgrid(self.xy, self.xy)

    def generate_ground_truth_image_stack(self, n_images, mask_radius=1.0):
        """
        Build a ground truth set of images, and associated binary mask of that
        image.

        :param n_images: Number of images to generate.
        :param mask_radius: used to build a binary mask derived from the ground
                            truth image.
        :return: ground truth images and associated mask.
        """

        # generate random positions
        positions = np.random.uniform(self.bump * self.sigma,
                                      self.n_xy - self.bump * self.sigma, (self.n_peaks,
                                                                           2,
                                                                           n_images))

        # make arrays to store images
        ground_truth_images = np.zeros((n_images, self.n_xy, self.n_xy))
        ground_truth_mask = np.zeros((n_images, self.n_xy, self.n_xy))
        threshold_for_mask = np.exp(-mask_radius ** 2.0 / 2.0)

        for peak in range(self.n_peaks):
            mux = positions[peak, 0, :]
            muy = positions[peak, 1, :]
            for mm in range(n_images):
                quadratic_form = (self.X - mux[mm]) ** 2.0 + (self.Y - muy[mm]) ** 2.0
                quadratic_form = quadratic_form / (2.0 * self.sigma ** 2.0)
                tmp_image = np.exp(-quadratic_form)
                mask_sel = tmp_image >= threshold_for_mask
                mask_image = np.zeros((self.n_xy, self.n_xy))
                mask_image[mask_sel] = 1.0
                ground_truth_images[mm, :, :] += tmp_image
                ground_truth_mask[mm, :, :] += mask_image
        sel = ground_truth_mask > 0.5
        ground_truth_mask[sel] = 1.0
        return ground_truth_images, ground_truth_mask

    def generate_data_with_normal_noise(self, n_images, snr=0.5,
                                        noise_sigma=4.0, noise_base=100.0,
                                        mask_radius=1.0,
                                        normalize='linear_scale'):
        """
        Build a dataset with uniform noise at a set level

        :param n_images: The number of
        :param snr: Signal to noise ratio
        :param noise_sigma: standard deviation of the normal noise (the
        `                   Gaussian sigma)
        :param noise_base: the base noise level (mean of gauss)
        :param mask_radius: The mask radius for ground_truth mask building
        :param normalize: scales noisy array
                          -if linear_scale, linearly scales to interval [-1,1]

        :return: ground truth image, mask and noisy images, class images
        """
        peak_value = snr * noise_sigma
        gt_img, gt_mask = self.generate_ground_truth_image_stack(n_images,
                                                                 mask_radius)
        noise = np.random.normal(noise_base, noise_sigma, gt_img.shape)

        # TODO: CLEAN THIS UP. WE DON'T NEED THIS
        # I want to make 'class' images as well.
        gt_class = np.zeros((n_images, 2, self.n_xy, self.n_xy))
        gt_class[:, 0, :, :] = 1.0 - gt_mask
        gt_class[:, 1, :, :] = gt_mask

        gt_noise = peak_value * gt_img + noise
        gt_noise_norm = None
        if normalize == 'linear_scale':
            a = np.min(gt_noise)
            b = np.max(gt_noise)
            c = 0
            d = 1.0
            gt_noise_norm = c + (d - c) * (gt_noise - a) / (b - a)

        return gt_img, gt_mask, gt_noise, gt_noise_norm, gt_class


class MixedNoiseDataMaker(object):
    """
        A class that can be used to build training data for 2D peak picking
        classification. The returned data is a stack of 2D images consisting of
        noisy data, error free data and a class labels. The noise levels change
        for each peak.
        """

    def __init__(self,
                 n_peaks=1,
                 sigma=1.26,
                 n_xy=128,
                 bump=0.0):
        """
            Initializes a class that generates a stack of two_d images of
            randomly placed Gaussian shaped peaks.

            :param n_peaks: The number of peaks to be placed in the unit box.
            :param sigma: The width of the gaussian peak.
            :param n_xy: The number of pixels in the unit box.
            :param bump: Controls border bumper to avoid placing peaks too
                         close to edge.
            """

        self.n_peaks = n_peaks
        self.n_xy = n_xy
        self.sigma = sigma
        self.bump = bump
        self.xy = np.linspace(0, n_xy - 1, n_xy)
        self.X, self.Y = np.meshgrid(self.xy, self.xy)
        self.snr_brackets = None
        self.set_snr_brackets()

    def set_snr_brackets(self,
                         snr_brackets=None
                         ):
        """
        Set the brackets of the peak signal to noise levels for each peak.

        :param snr_brackets: An array of psnr values.

        :return: void.
        """
        if snr_brackets is None:
            snr_brackets = [(1.5, 3.0),
                            (3.0, 5.0),
                            (5.0, 10.0)]
        self.snr_brackets = snr_brackets

    def generate_ground_truth_image_stack(self, n_images, mask_radius=1.0):
        """
            Build a ground truth set of images, and associated binary mask of
            that image.

            :param n_images: Number of images to generate.
            :param mask_radius: used to build a binary mask derived from the
                                ground truth image.
            :return: ground truth images stacks and associated mask stacks.
                     These stacks need to be further processed down to get
                     actual images.
            """

        # generate random positions
        positions = np.random.uniform(self.bump * self.sigma,
                                      self.n_xy - self.bump * self.sigma, (self.n_peaks,
                                                                           2,
                                                                           n_images))

        # make arrays to store images
        ground_truth_images = np.zeros((n_images, self.n_peaks, self.n_xy, self.n_xy))
        ground_truth_mask = np.zeros((n_images, self.n_peaks, self.n_xy, self.n_xy))
        threshold_for_mask = np.exp(-mask_radius ** 2.0 / 2.0)

        for mm in range(n_images):  # loop over images
            for peak in range(self.n_peaks):  # loop over peaks
                mux = positions[peak, 0, mm]
                muy = positions[peak, 1, mm]

                quadratic_form = (self.X - mux) ** 2.0 + (self.Y - muy) ** 2.0
                quadratic_form = quadratic_form / (2.0 * self.sigma ** 2.0)
                tmp_image = np.exp(-quadratic_form)

                mask_sel = tmp_image >= threshold_for_mask
                mask_image = np.zeros((self.n_xy, self.n_xy))
                mask_image[mask_sel] = 1  # the mask image now contains the peak number
                ground_truth_images[mm, peak, :, :] = tmp_image
                ground_truth_mask[mm, peak, :, :] = mask_image

        return ground_truth_images, ground_truth_mask

    def generate_data_with_normal_noise(self,
                                        n_images,
                                        snr_brackets=None,
                                        noise_sigma=4.0,
                                        noise_base=100.0,
                                        mask_radius=1.0,
                                        normalize='linear_scale'):
        """
            Build a dataset with uniform noise at a set level

            :param n_images: The number of images to generate
            :param snr_brackets: Signal to noise ratio
            :param noise_sigma: standard deviation of the normal noise (the
                                Gaussian sigma)
            :param noise_base: the base noise level (mean of gauss)
            :param mask_radius: The mask radius for ground_truth mask building
            :param normalize: scales noisy array
                            -if linear_scale, linearly scales to interval [0,1]
                            using range obtained from whole dataset.

            :return: ground truth image, mask / class, noisy images, scaled
                     noise images
            """

        # first we set the snr brackets
        if snr_brackets is not None:
            self.set_snr_brackets(snr_brackets=snr_brackets)

        # now we generate images stacks
        # note, these images still need to be projected onto a single frame
        # with proper weights

        gt_img_stack, gt_mask_stack = self.generate_ground_truth_image_stack(n_images,
                                                                             mask_radius)

        gt_imgs = np.zeros((n_images, self.n_xy, self.n_xy))
        gt_msks = np.zeros((n_images, self.n_xy, self.n_xy))
        gt_noise = np.zeros((n_images, self.n_xy, self.n_xy))

        for mm in range(n_images):
            img_stack = gt_img_stack[mm, :, :, :]
            msk_stack = gt_mask_stack[mm, :, :, :]
            gt_img = np.zeros((self.n_xy, self.n_xy))

            # get levels
            levels = np.arange(0, len(self.snr_brackets))
            these_levels = np.random.choice(levels, self.n_peaks, replace=True)

            # get peak values, we multiply these values with the gaussian peaks we have
            peak_values = []
            for pp, this_level in enumerate(these_levels):
                snr_bracket = self.snr_brackets[this_level]
                snr = np.random.uniform(snr_bracket[0], snr_bracket[1], 1)[0]
                peak_value = snr * noise_sigma
                peak_values.append(peak_value)
                tmp_img = img_stack[pp, :, :]
                gt_img += tmp_img * peak_value
                msk_stack[pp, :, :] = msk_stack[pp, :, :] * (1 + this_level)

            # looped through all levels
            gt_imgs[mm, :, :] = gt_img
            gt_msks[mm, :, :] = np.max(msk_stack, axis=0)
            gt_noise[mm, :, :] = gt_img + np.random.normal(noise_base,
                                                           noise_sigma,
                                                           gt_img.shape)

        gt_noise_norm = None
        if normalize == 'linear_scale':
            a = np.min(gt_noise)
            b = np.max(gt_noise)
            c = 0
            d = 1.0
            gt_noise_norm = c + (d - c) * (gt_noise - a) / (b - a)
            # gt_noise -= np.min(gt_noise)
            # gt_noise /= np.max(gt_noise)
        return gt_imgs, gt_msks, gt_noise, gt_noise_norm


def tst(show=True, n_xy=1000, n_imgs=10):
    """
    Provides a simple check if the peaks are generated
    Also serves as an indication how to use the above class.

    :param show: If True 3 sample images will be shown.
    :param n_xy: Dimensions of the box
    :param n_imgs: Number of images generated
    :return: True is test is passed
    """
    n_peaks = 3
    sigma = 1.28
    img_engine = DataMaker(n_peaks, sigma=sigma, n_xy=n_xy)
    imgs, msks, n_imgs, norm_img, n_class = img_engine.generate_data_with_normal_noise(n_imgs,
                                                                                       snr=0.5)
    img_sum = np.sum(imgs, axis=(1, 2))
    img_sum = np.mean(img_sum)
    theory = 2.0 * np.pi * sigma * sigma * n_peaks * n_xy * n_xy
    residual = abs(theory - img_sum) / theory

    if show:
        for img, msk, nimg, cimg in zip(imgs[:3], msks[:3], n_imgs[:3], n_class[:3]):
            plt.figure(figsize=(25, 5))
            plt.subplot(151)
            plt.imshow(img)
            plt.colorbar(shrink=0.70)
            plt.title('Ground Truth')

            plt.subplot(152)
            plt.imshow(msk)
            plt.colorbar(shrink=0.70)
            plt.title('Mask')

            plt.subplot(153)
            plt.imshow(nimg)
            plt.colorbar(shrink=0.70)
            plt.title('Noisy Image')

            plt.subplot(154)
            plt.imshow(cimg[0, :, :])
            plt.colorbar(shrink=0.70)
            plt.title('bg class')

            plt.subplot(155)
            plt.imshow(cimg[1, :, :])
            plt.colorbar(shrink=0.70)
            plt.title('peak class')

            plt.show()

    if not show:
        if residual > 1e-2:
            print(residual)
        assert (residual < 1e-2)
        return True

    return None


def tst_mixed(show=True, n_xy=32, n_imgs=10):
    """
    Displays data for mixed noise data.
    """
    n_peaks = 3
    sigma = 1.28
    img_engine = MixedNoiseDataMaker(n_peaks, sigma=sigma, n_xy=n_xy)
    imgs, msks, n_imgs, norma_imgs = img_engine.generate_data_with_normal_noise(n_imgs)

    if show:
        for img, msk, nimg in zip(imgs, msks, norma_imgs):
            plt.figure(figsize=(17, 5))
            plt.subplot(131)

            plt.imshow(img)
            plt.colorbar(shrink=0.80)
            plt.title('Ground Truth')

            plt.subplot(132)
            plt.imshow(msk)
            plt.colorbar(shrink=0.80)
            plt.title('Mask')

            plt.subplot(133)
            plt.imshow(nimg)
            plt.colorbar(shrink=0.80)
            plt.title('Noisy Image')

            plt.show()

    return None


def run_tst(n_xy=1000, n_imgs=1):
    tst(False, n_xy, n_imgs)


if __name__ == "__main__":
    # run_tst()
    # tst(True, 32, 10)
    tst_mixed(True, 32, 20)
