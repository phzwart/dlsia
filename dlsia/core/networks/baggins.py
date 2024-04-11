import numpy as np
import torch
from torch import nn

def arithmetic_mean_std(results):
    """
    Computes Arithmatic mean and standard deviation

    Parameters
    ----------
    results : an tensor with results. we average over channel 1

    Returns
    -------
    mean and standard deviation (arithmatic)

    """
    m = torch.mean( results, dim = 1)
    s = torch.std(results, dim=1)
    return m, s

def geometric_mean_std(results, eps=1e-12):
    """
    Computes geometric mean and standard deviation

    Parameters
    ----------
    results : an tensor with results. we average over channel 1

    Returns
    -------
    mean and standard deviation (geometric)

    """
    m = torch.mean( torch.log(results+eps), dim=1 )
    m = torch.exp(m)
    s = torch.std(torch.log(results+eps), dim=1)
    s = torch.exp(s)
    return m,s

class model_baggin(nn.Module):
    """
    Bagg a number of models
    """

    def __init__(self, models, model_type='classification', returns_normalized=False, average_type="arithmetic"):
        """
        Bag a number of models together and get a mean and std estimate

        Parameters
        ----------
        models : a list of neural networks
        model_type : regression or classification
        returns_normalized : if False and added softmax will be performed if model_type is classification
        """
        super(model_baggin, self).__init__()

        self.models = models
        self.model_type = model_type
        self.returns_normalized = returns_normalized
        if average_type == "geometric":
            self.mean_std_calculator = geometric_mean_std
        else:
            self.mean_std_calculator = arithmetic_mean_std

        assert self.model_type in ["regression", "classification"]

    def forward(self, x, device="cpu", return_std=False):
        """
        Standard forward model

        Parameters
        ----------
        x : input tensor
        device : where will we do the calculations?
        return_std: If true the standard deviation will be returned

        Returns
        -------
        mean and standard deviation
        """
        mean = 0
        std = 0
        N = 0
        results = []
        x = x.to(device)
        with torch.no_grad():
            for model in self.models:
                N += 1
                if device != "cpu":
                    torch.cuda.empty_cache()
                tmp_result = model.to(device)(x)#.cpu()
                if self.model_type == "classification":
                    if not self.returns_normalized:
                        tmp_result = nn.Softmax(dim=1)(tmp_result)
                results.append(tmp_result.unsqueeze(1).cpu())
                #model.cpu()
            results = torch.cat(results, dim=1)
            mean, std = self.mean_std_calculator(results)

            if self.model_type == "classification":
                norma = torch.sum(mean, dim=1).unsqueeze(1)
                mean = mean / norma
                std = std / norma
            if not return_std:
                return mean
            return mean, std / np.sqrt(N)


def estimate_quantiles_with_bootstrap_adaptive(tensor, quantiles, num_samples=100, noise_scale=0.01):
    """
    Estimate quantiles for each element across the axis C of a tensor using a smooth bootstrap approach.
    The noise scale is adaptive, based on the standard deviation of each channel, scaled by the provided noise_scale.

    :param tensor: Input tensor of shape N, C, Y, X or N, C, Z, Y, X.
    :param quantiles: A list of quantiles to compute, each a float between 0 and 1.
    :param num_samples: Number of bootstrap samples to generate.
    :param noise_scale: Scaling factor for the standard deviation-based noise.
    :return: A tensor of quantiles with one less dimension than the input (quantiles replace the C dimension).
    """
    # Validate tensor dimensionality
    assert tensor.dim() in [4, 5], "Input tensor should have 4 or 5 dimensions (N, C, Y, X or N, C, Z, Y, X)"

    # Determine the shape for the output tensor, excluding the 'C' dimension
    output_shape = (len(quantiles), *tensor.shape[0:1], *tensor.shape[2:])

    # Initialize tensor to store bootstrap sample quantiles
    bootstrap_quantiles = torch.zeros((num_samples, *output_shape), dtype=tensor.dtype, device=tensor.device)

    # Calculate standard deviation for each channel
    stdevs = torch.std(tensor, dim=(0, *range(2, tensor.dim())), keepdim=True)

    for i in range(num_samples):
        # Generate noise based on the channel-wise standard deviation
        noise = torch.randn_like(tensor) * stdevs * noise_scale
        noisy_tensor = tensor + noise

        # Compute and store quantiles for the noisy tensor
        bootstrap_quantiles[i] = torch.quantile(noisy_tensor, torch.tensor(quantiles, dtype=tensor.dtype, device=tensor.device), dim=1, keepdim=False)

    # Aggregate the quantiles across all bootstrap samples
    final_quantiles = torch.mean(bootstrap_quantiles, dim=0)

    return final_quantiles


class model_baggin_univariate_quantiles(nn.Module):
    """
    Bagg a number of models
    """

    def __init__(self, models, quantiles=[0.05, 0.10, 0.25, 0.50,]):
        """
        Bag a number of models together and get estimates of the qunatiles

        Parameters
        ----------
        models : a list of neural networks
        model_type : regression or classification
        returns_normalized : if False and added softmax will be performed if model_type is classification
        """
        super(model_baggin, self).__init__()

        self.models = models
        self.quantiles = quantiles

    def forward(self, x, device="cpu"):
        """
        Standard forward model

        Parameters
        ----------
        x : input tensor
        device : where will we do the calculations?
        return_std: If true the standard deviation will be returned

        Returns
        -------
        mean and standard deviation
        """
        mean = 0
        std = 0
        N = 0
        results = []
        x = x.to(device)
        with torch.no_grad():
            for model in self.models:
                N += 1
                if device != "cpu":
                    torch.cuda.empty_cache()
                tmp_result = model.to(device)(x)#.cpu()
                results.append(tmp_result.unsqueeze(1).cpu())
            results = torch.cat(results, dim=1)




class autoencoder_labeling_model_baggin(nn.Module):
    """
    Bagg a number of models
    """

    def __init__(self, models, returns_normalized=False):
        """
        Bag a number of models together and get a mean and std estimate.
        To be used for bagging SparseNet.SparseAEC models.

        Parameters
        ----------
        models : a list of SparseNet.SparseAEC neural networks
        returns_normalized : if False added softmax will be performed
        """
        super(autoencoder_labeling_model_baggin, self).__init__()

        self.models = models
        self.returns_normalized = returns_normalized

    def forward(self, x, device="cpu", return_std=False):
        """
        Standard forward model

        Parameters
        ----------
        x : input tensor
        device : where will we do the calculations?
        return_std: If True standard deviations will be returned

        Returns
        -------
        mean and standard deviation
        """
        mean_class = 0
        std_class = 0

        mean_recon = 0
        std_recon = 0

        N = 0
        with torch.no_grad():
            for model in self.models:
                N += 1
                if device != "cpu":
                    torch.cuda.empty_cache()
                tmp_recon, tmp_class = model.to(device)(x.to(device))
                tmp_recon = tmp_recon.cpu()
                tmp_class = tmp_class.cpu()

                # average classifications

                if not self.returns_normalized:
                    tmp_class = nn.Softmax(dim=1)(tmp_class)
                mean_class += tmp_class
                std_class += tmp_class ** 2.0

                # average reconstructions
                mean_recon += tmp_recon
                std_recon += tmp_recon ** 2.0

                model.cpu()
            mean_class = mean_class / N
            std_class = std_class / (N - 1)
            std_class = torch.sqrt(std_class - mean_class * mean_class) / np.sqrt(N)
            norma = torch.sum(mean_class, dim=1).unsqueeze(1)
            mean_class = mean_class / norma
            std_class = std_class / norma

            mean_recon = mean_recon / N
            std_recon = std_recon / (N - 1)
            std_recon = torch.sqrt(std_recon - mean_recon * mean_recon) / np.sqrt(N)

            if not return_std:
                return mean_recon, mean_class
            return mean_recon, std_recon, mean_class, std_class
