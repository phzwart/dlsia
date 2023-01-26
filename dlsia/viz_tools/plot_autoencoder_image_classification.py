import matplotlib.pyplot as plt


def plot_autoencoder_and_label_results(input_img,
                                       output_img,
                                       p_classification,
                                       class_names):
    """

    Parameters
    ----------
    input_img : Input image
    output_img : Autoencoder output image
    p_classification : class probabilities
    class_names : class names

    Returns
    -------
    A matplotlib figure

    """

    shrink = 0.76
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 2.5))
    tmp = ax[0].imshow(input_img, cmap='viridis')
    cbar = fig.colorbar(tmp, ax=ax[0], shrink=shrink)
    ax[0].set_title("Input Image")
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    tmp = ax[1].imshow(output_img, cmap='viridis')
    cbar = fig.colorbar(tmp, ax=ax[1], shrink=shrink)
    ax[1].set_title("Output Image")
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    tmp = ax[2].bar(class_names, p_classification)
    ax[2].set_title("Class Probabilities")
    return fig


def plot_autoencoder_and_label_results_with_std(input_img,
                                                output_img,
                                                std_img,
                                                p_classification,
                                                std_p_classification,
                                                class_names):
    """

    Parameters
    ----------
    input_img : Input image
    output_img : Autoencoder output image
    std_img : reconstruction standard deviations
    p_classification : class probabilities
    std_p_classification : class probabilities standard deviations
    class_names : class names

    Returns
    -------
    A matplotlib figure

    """

    shrink = 0.76
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 2.0))
    tmp = ax[0].imshow(input_img, cmap='viridis')
    cbar = fig.colorbar(tmp, ax=ax[0], shrink=shrink)
    ax[0].set_title("Input Image")
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    tmp = ax[1].imshow(output_img, cmap='viridis')
    cbar = fig.colorbar(tmp, ax=ax[1], shrink=shrink)
    ax[1].set_title("Output Image")
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    tmp = ax[2].imshow(std_img, cmap="RdBu_r")
    cbar = fig.colorbar(tmp, ax=ax[2], shrink=shrink)
    ax[2].set_title("Standard Deviation")
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    tmp = ax[3].bar(class_names,
                    p_classification,
                    yerr=std_p_classification,
                    align='center', ecolor='black', capsize=5
                    )
    ax[3].set_title("Class Probabilities")
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.34,
                        hspace=0.4)

    return fig


def plot_image_and_class_probabilities(input_img,
                                       p_classification,
                                       class_names,
                                       std_p_classification=None):
    """

    Parameters
    ----------
    input_img : Input numpy image
    p_classification : classification probabilities
    class_names : class names
    std_p_classification : stdev of probabilities if available.

    Returns
    -------
    A matplotlib image
    """

    shrink = 0.95
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 2.5))
    tmp = ax[0].imshow(input_img, cmap='viridis')
    cbar = fig.colorbar(tmp, ax=ax[0], shrink=shrink)
    ax[0].set_title("Input Image")
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    tmp = ax[1].bar(class_names,
                    p_classification,
                    yerr=std_p_classification,
                    align='center', ecolor='black', capsize=5
                    )
    ax[1].set_title("Class Probabilities")
    return fig
