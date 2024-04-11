import torch
from dlsia.core.utils import blur

def downsample_3D_image(image, factor, device='cpu'):
    """
    Downsamples a 3D image using Gaussian blurring and subsampling.

    First, the image is blurred using a 3D Gaussian filter to prevent aliasing.
    Then, it is subsampled by selecting every nth voxel where n is the downsampling factor.

    Args:
        image (torch.Tensor): The 3D image to be downsampled, with dimensions (N, C, D, H, W).
        factor (int): The downsampling factor.
        device (str, optional): The device to perform computations on. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The downsampled 3D image.
    """
    sigma = factor / 2.0
    max_kernel_size = 2 * (int(5 * sigma) // 2) + 1
    blurrer = blur.GaussianBlur3D(sigma, max_kernel_size=max_kernel_size).to(device)
    slicer = slice(factor // 2-1, -1, factor)

    with torch.no_grad():
        blurred_image = blurrer(image.to(device))
        # Subsampling the image
        blurred_image = blurred_image[:, :, slicer, slicer, slicer]
    return blurred_image

def downsample_3D_labels(class_map, num_classes, factor, missing_label=-1, device='cpu', missing_fraction=.05):
    """
    Downsamples a 3D label map, ensuring that the downsampling process does not bias
    towards the background or missing label in sparse label scenarios.

    This function first converts the class labels into a one-hot encoding format, applies 3D downsampling,
    and then converts it back, prioritizing actual labels over the missing label unless the entire
    region is missing.

    Args:
        class_map (torch.Tensor): The 3D tensor of class labels to be downsampled, with dimensions (N, D, H, W).
        num_classes (int): The number of different classes in the class_map, excluding the missing label.
        factor (int): The downsampling factor.
        missing_label (int, optional): The label used for missing data. Defaults to -1.
        device (str, optional): The device for computations. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The downsampled class map.
    """
    N, Z, Y, X = class_map.shape
    # Adjusting for an extra channel to represent the missing label
    one_hot_map = torch.zeros(N, num_classes + 1, Z, Y, X, device=device)
    one_hot_map[:, 0, :, :, :][class_map == missing_label] = 1
    for i in range(1, num_classes + 1):
        one_hot_map[:, i, :, :, :][class_map == i - 1] = 1
    ds_ohm = downsample_3D_image(one_hot_map, factor, device)
    relevant_labels = ds_ohm[:, 1:, :, :, :]
    ds_class_map = torch.argmax(relevant_labels, dim=1)
    # Adjust for the fact that we excluded the first channel
    ds_class_map[ds_ohm[:, 0, :, :, :] >= 1-missing_fraction] = missing_label
    return ds_class_map
