import numpy as np
import torch
from torch import nn
import torch.distributions as dist

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

def generate_moments(results):
    """
    Computes four centralized moments

    Parameters
    ----------
    results : an tensor with results. we average over channel 1

    Returns
    -------
    The first 4 centralized moments,

    """
    m = torch.mean( results, dim = 1)
    s = torch.std(results, dim=1)
    return m, s

def calculate_skewness_and_kurtosis(data):
    # Calculate mean and standard deviation along dimension 1
    mean = torch.mean(data, dim=1, keepdim=True)
    std_dev = torch.std(data, dim=1, unbiased=True, keepdim=True)

    # Calculate third and fourth central moments
    deviations = data - mean
    third_moment = torch.mean(deviations**3, dim=1)
    fourth_moment = torch.mean(deviations**4, dim=1)

    # Correcting the bias in skewness and kurtosis
    n = data.size(1)
    skewness = (n / ((n - 1) * (n - 2))) * third_moment / (std_dev.squeeze(1)**3)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * fourth_moment / (std_dev.squeeze(1)**4) - (3 * (n - 1)**2 / ((n - 2) * (n - 3)))

    return mean.squeeze(1), std_dev.squeeze(1), skewness, kurtosis

def cornish_fisher_expansion(data, quantiles=[0.05, 0.95]):
    mean, std, skewness, kurtosis = calculate_skewness_and_kurtosis(data)

    # Calculate the Cornish-Fisher expansion for each quantile
    cf_quantiles = []
    for q in quantiles:
        z = dist.Normal(0, 1).icdf(torch.tensor(q))  # Get the z-score using PyTorch's distributions

        # Calculate the adjusted z-score using the Cornish-Fisher expansion
        z_adj = z + (1/6) * (z**2 - 1) * skewness + (1/24) * (z**3 - 3*z) * kurtosis - (1/36) * (2*z**3 - 5*z) * skewness**2

        # Convert the adjusted z-score to the actual quantile value in the data distribution
        cf_quantile = mean + z_adj * std
        cf_quantiles.append(cf_quantile)

    return mean, std, torch.stack(cf_quantiles, dim=1)  # Return stacked results


class model_baggin(nn.Module):
    """
    Bagg a number of models
    """

    def __init__(self, models,
                 model_type='classification',
                 returns_normalized=False,
                 average_type="arithmatic",
                 quantiles=[0.05, 0.95]
                 ):
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

        self.quantiles = quantiles



        assert self.model_type in ["regression", "classification"]

    def forward(self, x, device="cpu", return_std=False, return_quantile=False):
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
            if return_std:
                mean, std = self.mean_std_calculator(results)
            if return_quantile:
                mean,std,quantiles = cornish_fisher_expansion(data=results, quantiles=self.quantiles)
                return mean, std, quantiles

            if self.model_type == "classification":
                if not return_quantile:
                    norma = torch.sum(mean, dim=1).unsqueeze(1)
                    mean = mean / norma
                    std = std / norma
                    if not return_std:
                        return mean
                    return mean, std / np.sqrt(N)
                else:
                    print("HOPPA")
                    mean,std,quantiles = cornish_fisher_expansion(data=results,
                                                                  quantiles=self.quantiles)
                    return mean, std, quantiles


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
