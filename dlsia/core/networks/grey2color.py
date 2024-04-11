import torch
from torch import nn
from einops import rearrange


# Encoder Module
def make_oder(dim_list):
    layers = []
    for i in range(len(dim_list) - 1):
        layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(dim_list[i + 1]))
    return nn.Sequential(*layers)

# Decoder Module
def make_decoder(dim_list):
    layers = []
    for i in range(len(dim_list) - 1):
        layers.append(nn.Linear(dim_list[i + 1], dim_list[i]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(dim_list[i]))
    return nn.Sequential(*layers)

# Autoencoder Class
class Grey2Color(nn.Module):
    def __init__(self, channel_dims, encoder_dims=[256, 128], decoder_dims=[128, 256], _latent_dim=3):
        super(Grey2Color, self).__init__()

        # Input channel dimension to encoder should match channel_dims
        self._latent_dim =_latent_dim
        self._encoder_dims = encoder_dims
        self._decoder_dims = decoder_dims
        self.channel_dims = channel_dims

        self.encoder_dims = [self.channel_dims] + self._encoder_dims + [self._latent_dim]

        # Output channel dimension from decoder should match channel_dims
        self.decoder_dims = [self._latent_dim] + self._decoder_dims + [self.channel_dims]

        self.encoder = make_oder(self.encoder_dims)
        self.decoder = make_oder(self.decoder_dims)


    def forward(self, x):
        # Rearrange from N C Y X to (N Y X) C
        N,C,Y,X = x.shape
        x = rearrange(x, 'N C Y X -> (N Y X) C ')

        # Encoding
        z = torch.sigmoid(self.encoder(x))

        # Decoding
        x_recon = self.decoder(z)

        # Rearrange back from (N Y X) C to N C Y X
        x_recon = rearrange(x_recon, '(N Y X) C -> N C Y X', N=N,Y=Y,X=X )

        return x_recon

    def histogram_equalization(self, x):
        # Calculate histogram
        hist = torch.histc(x, bins=256, min=0, max=1)

        # Compute the cumulative distribution function (CDF)
        cdf = torch.cumsum(hist, dim=0)
        cdf_min = cdf.min()

        # Normalize CDF
        cdf_normalized = (cdf - cdf_min) / (cdf.max() - cdf_min)

        # Map original values to equalized values
        bin_edges = torch.linspace(0, 1, 257)
        equalized = torch.zeros_like(x)

        for i in range(256):
            mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
            equalized[mask] = cdf_normalized[i]
        return equalized


    def latent_image(self, x, scaling_method='hist_eq'):
        with torch.no_grad():
            N, C, Y, X = x.shape

            # Rearrange from N C Y X to (N Y X) C
            x = rearrange(x, 'N C Y X -> (N Y X) C')

            # Encoding
            z = torch.sigmoid(self.encoder(x))

            if scaling_method == 'sigmoid':
                # Apply sigmoid to rescale to [0, 1]
                z = torch.sigmoid(z)
            elif scaling_method == 'minmax':
                # Apply min-max scaling to [0, 1]
                z_min = z.min(dim=0, keepdim=True)[0]
                z_max = z.max(dim=0, keepdim=True)[0]
                z = (z - z_min) / (z_max - z_min + 1e-10)
            elif scaling_method == 'hist_eq':
                # Apply histogram equalization
                z = self.histogram_equalization(z)
            else:
                raise ValueError("Invalid scaling method. Choose 'sigmoid', 'minmax', or 'hist_eq'.")

            # Rearrange back from (N Y X) C to N Y X C
            z = rearrange(z, '(N Y X) C -> N Y X C', N=N, Y=Y, X=X)

            return z
