import matplotlib.pyplot as plt
import numpy as np


def build_latent_space_image_viewer(images,
                                    latent_2d,
                                    n_bins=20,
                                    mode="nearest",
                                    min_count=1,
                                    max_count=20,
                                    size=(10, 10)):
    """
    Visualize latent space with either mean of examplar images.

    Parameters
    ----------
    images : input 2D images (N Y X), numpy array
    latent_2d : the vectors in latent space (N, Latent_dim)
    n_bins : number of bins in X and Y direction
    mode : "nearest" or "mean". When nearest, the image nearest to the center
            of the bin is displayed, otherwise the mean
    min_count : ignore bins with counts below this number
    max_count : if number of items in bin above this value, the transparency
                will be zero.
    size : the size of the output image

    Returns
    -------
    a matplotlib fig object

    """
    assert mode in ["nearest", "mean"]

    bin_width = 1.0 / (n_bins - 3)
    minx = np.min(latent_2d[:, 0])
    miny = np.min(latent_2d[:, 1])
    maxx = np.max(latent_2d[:, 0])
    maxy = np.max(latent_2d[:, 1])

    dx = (maxx - minx) * bin_width
    dy = (maxy - miny) * bin_width
    dxy = np.array([[dx, dy]])
    start = np.array([[minx - dx / 2.0, miny - dy / 2.0]])

    indx = (latent_2d - start) / dxy
    max_indx = (np.max(indx, axis=0) + 0.5).astype(int)
    min_indx = (np.min(indx, axis=0) + 0.5).astype(int)

    int_indx = (indx + 0.5).astype(int)

    fig, axs = plt.subplots(nrows=n_bins, ncols=n_bins)
    plt.gcf().set_size_inches(size)

    for ii in range(n_bins):
        for jj in range(n_bins):
            selx = int_indx[:, 0] == ii
            sely = int_indx[:, 1] == jj
            sel = selx & sely
            selected_images = images[sel, ...]
            sel_indx = indx[sel, ...]
            ncount = selected_images.shape[0]
            if ncount >= min_count:
                if mode == "nearest":
                    this_one = (np.abs(sel_indx) - np.array([[ii, jj]]))
                    this_one = np.argmin(np.sum(this_one * this_one, axis=-1))
                    this_img = selected_images[this_one, ...]

                if mode == "mean":
                    this_img = np.mean(selected_images, axis=0)

                alpha = 1.0
                if ncount < max_count:
                    alpha = (ncount - min_count) / (max_count - min_count)
                axs[-(jj + 1), ii].imshow(this_img, alpha=alpha)
                axs[-(jj + 1), ii].set_axis_off()
            else:
                axs[-(jj + 1), ii].set_axis_off()
    return fig
