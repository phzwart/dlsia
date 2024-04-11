import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur3D(nn.Module):
    """
    Apply 3D Gaussian blurring across multiple channels and batch sizes.

    This class extends Gaussian blurring to 3D tensors and is designed to work
    with inputs having multiple channels and batch sizes, applying the same sigma
    (standard deviation) across all channels.

    Attributes:
        sigma (nn.Parameter): The standard deviation of the Gaussian kernel.
        max_kernel_size (int): The maximum size of the Gaussian kernel.
        kernel (Tensor): The computed Gaussian kernel.

    Args:
        initial_sigma (float): The initial value of the standard deviation for the Gaussian kernel.
        max_kernel_size (int, optional): The maximum size of the Gaussian kernel. Defaults to 11.
    """
    def __init__(self, initial_sigma, max_kernel_size=11):
        super(GaussianBlur3D, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma], dtype=torch.float32))
        self.max_kernel_size = 2*int(max_kernel_size//2)+1

        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, self.sigma)

    def create_gaussian_kernel(self, kernel_size, sigma, device='cpu'):
        """
        Creates a normalized 3D Gaussian kernel.

        Args:
            kernel_size (int): The size of the kernel.
            sigma (Tensor): The standard deviation for the Gaussian kernel.
            device (str, optional): The device to create the kernel on. Defaults to 'cpu'.

        Returns:
            Tensor: The 3D Gaussian kernel.
        """
        range = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        x = range.view(-1, 1, 1).repeat(1, kernel_size, kernel_size).to(device)
        y = range.view(1, -1, 1).repeat(kernel_size, 1, kernel_size).to(device)
        z = range.view(1, 1, -1).repeat(kernel_size, kernel_size, 1).to(device)

        gaussian_kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x):
        """
        Apply Gaussian blurring to the input tensor.

        Args:
            x (Tensor): The input tensor to be blurred.

        Returns:
            Tensor: The blurred tensor.
        """
        sigma = self.sigma.abs() + 1e-6
        device = x.device
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, sigma, device).to(device)

        padding = int((self.max_kernel_size - 1) / 2)
        x_padded = F.pad(x, [padding] * 6, mode='reflect')

        kernel = self.kernel.repeat(x.size(1), 1, 1, 1, 1)
        blurred = F.conv3d(x_padded, kernel, groups=x.size(1), padding=0)
        return blurred

class GaussianBlur2D(nn.Module):
    """
    Apply 2D Gaussian blurring across multiple channels and batch sizes.

    This class provides a mechanism to apply Gaussian blurring to 2D tensors (e.g., images),
    supporting multiple channels (e.g., RGB images) and various batch sizes. The same standard
    deviation (sigma) is applied across all channels to maintain consistency in blurring effect.

    Attributes:
        sigma (nn.Parameter): The standard deviation of the Gaussian kernel.
        max_kernel_size (int): The maximum size of the kernel used for blurring.
        kernel (Tensor): The computed Gaussian kernel tensor.

    Args:
        initial_sigma (float): The initial standard deviation value for the Gaussian kernel.
        max_kernel_size (int, optional): The maximum dimension of the Gaussian kernel. Defaults to 11.
    """
    def __init__(self, initial_sigma, max_kernel_size=11):
        super(GaussianBlur2D, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([initial_sigma], dtype=torch.float32))
        self.max_kernel_size = max_kernel_size
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, self.sigma)

    def create_gaussian_kernel(self, kernel_size, sigma, device='cpu'):
        """
        Generates a normalized 2D Gaussian kernel using the specified sigma and kernel size.

        Args:
            kernel_size (int): The size of the kernel.
            sigma (Tensor): The sigma (standard deviation) value for the Gaussian distribution.
            device (str, optional): The device (CPU or GPU) where the kernel will be allocated. Defaults to 'cpu'.

        Returns:
            Tensor: A 2D tensor representing the normalized Gaussian kernel.
        """
        range = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        x = range.view(-1, 1).repeat(1, kernel_size).to(device)
        y = range.view(1, -1).repeat(kernel_size, 1).to(device)

        gaussian_kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        """
        Applies Gaussian blurring to the input tensor.

        This method pads the input tensor to handle borders properly, updates the Gaussian kernel if necessary,
        and then applies convolution to blur the input tensor.

        Args:
            x (Tensor): The input tensor to be blurred.

        Returns:
            Tensor: The blurred tensor, having the same dimensions as the input.
        """
        sigma = self.sigma.abs() + 1e-6
        device = x.device
        self.kernel = self.create_gaussian_kernel(self.max_kernel_size, sigma, device).to(device)

        padding = int((self.max_kernel_size - 1) / 2)
        x_padded = F.pad(x, [padding, padding, padding, padding], mode='reflect')

        kernel = self.kernel.repeat(x.size(1), 1, 1, 1)
        blurred = F.conv2d(x_padded, kernel, groups=x.size(1), padding=0)

        return blurred
