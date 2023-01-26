# TODO: add docstrings
import einops
import torch.nn as nn
from dlsia.core.networks import msdnet, tunet


class MSDEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 depth,
                 base_channels,
                 growth_rate,
                 latent_dimension,
                 max_dilation=10,
                 MSD_depth=10):

        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.latent_dimension = latent_dimension
        self.max_dilation = max_dilation
        self.MSD_depth = MSD_depth

        ysizing = tunet.unet_sizing_chart(N=self.input_shape[1], depth=self.depth, stride=2, kernel=3, dilation=1)
        xsizing = tunet.unet_sizing_chart(N=self.input_shape[2], depth=self.depth, stride=2, kernel=3, dilation=1)

        self.sizing_chart = [ysizing, xsizing]
        self.channel_sequence = [(self.input_channels, self.base_channels)]
        for ii in range(1, depth):
            self.channel_sequence.append(
                (self.channel_sequence[ii - 1][1], int(self.channel_sequence[ii - 1][1] * self.growth_rate)))

        self.final_image_shape = (self.channel_sequence[-1][1],
                                  self.sizing_chart[0]['Sizes'][self.depth - 1][self.depth][1],
                                  self.sizing_chart[1]['Sizes'][self.depth - 1][self.depth][1])
        self.network_operator_sequence = self.stack_blocks()
        self.final_operator = self.final_linear_layer()

    def build_block(self, from_depth):

        in_channels = self.channel_sequence[from_depth][0]
        out_channels = self.channel_sequence[from_depth][1]

        # MSDNet that sorts the channels out
        this_MSDNet = msdnet.MixedScaleDenseNetwork(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    num_layers=self.MSD_depth,
                                                    max_dilation=self.max_dilation,
                                                    activation=nn.ReLU(),
                                                    layer_width=self.max_dilation,
                                                    normalization=nn.BatchNorm2d)
        # MaxPool
        this_MaxPool = tunet.build_down_operator(chart=self.sizing_chart,
                                                 from_depth=from_depth,
                                                 to_depth=from_depth + 1,
                                                 conv_kernel=nn.MaxPool2d,
                                                 key="Pool_Settings")
        operator = nn.Sequential(this_MSDNet, this_MaxPool)
        return operator

    def stack_blocks(self):
        operators = []
        for ii in range(self.depth):
            operators.append(self.build_block(ii))
        operators = nn.Sequential(*operators)
        return operators

    def final_linear_layer(self):
        print()
        in_features = self.final_image_shape[0] * self.final_image_shape[1] * self.final_image_shape[2]
        out_features = self.latent_dimension
        operator = nn.Linear(in_features=in_features, out_features=out_features)
        return operator

    def forward(self, x):
        x = self.network_operator_sequence(x)
        x = einops.rearrange(x, "N C Y X -> N (C Y X)")
        x = self.final_operator(x)
        return x


class MSDDecoder(nn.Module):
    def __init__(self,
                 output_shape,
                 depth,
                 base_channels,
                 growth_rate,
                 latent_dimension,
                 max_dilation=10,
                 MSD_depth=10):
        super().__init__()
        self.output_shape = output_shape
        self.input_channels = output_shape[0]
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.latent_dimension = latent_dimension
        self.max_dilation = max_dilation
        self.MSD_depth = MSD_depth

        ysizing = tunet.unet_sizing_chart(N=self.output_shape[1], depth=self.depth, stride=2, kernel=3, dilation=1)
        xsizing = tunet.unet_sizing_chart(N=self.output_shape[2], depth=self.depth, stride=2, kernel=3, dilation=1)
        self.sizing_chart = [ysizing, xsizing]
        self.channel_sequence = [(self.input_channels, self.base_channels)]
        for ii in range(1, depth):
            self.channel_sequence.append(
                (self.channel_sequence[ii - 1][1], int(self.channel_sequence[ii - 1][1] * self.growth_rate)))
        self.first_image_shape = (self.channel_sequence[-1][1],
                                  self.sizing_chart[0]['Sizes'][self.depth - 1][self.depth][1],
                                  self.sizing_chart[1]['Sizes'][self.depth - 1][self.depth][1])
        self.first_operator = self.first_linear_layer()
        self.network_operator_sequence = self.stack_blocks()

    def first_linear_layer(self):
        in_features = self.latent_dimension
        out_features = self.first_image_shape[0] * self.first_image_shape[1] * self.first_image_shape[2]
        operator = nn.Linear(in_features=in_features, out_features=out_features)
        return operator

    def build_block(self, from_depth):

        out_channels = self.channel_sequence[from_depth][0]
        in_channels = self.channel_sequence[from_depth][1]

        # we now need to build the upscaling parameters
        this_convT = tunet.build_up_operator(chart=self.sizing_chart,
                                             from_depth=from_depth + 1,
                                             to_depth=from_depth,
                                             in_channels=in_channels,
                                             out_channels=in_channels,
                                             conv_kernel=nn.ConvTranspose2d,
                                             key="convT_settings")

        # MSD net part to sort out the channels
        this_MSDNet = msdnet.MixedScaleDenseNetwork(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    num_layers=self.MSD_depth,
                                                    max_dilation=self.max_dilation,
                                                    activation=nn.ReLU(),
                                                    layer_width=self.max_dilation,
                                                    normalization=nn.BatchNorm2d)
        operator = nn.Sequential(this_convT, this_MSDNet)
        print(in_channels, out_channels)
        return operator

    def stack_blocks(self):
        sequence = []
        for ii in range(0, self.depth)[::-1]:
            sequence.append(self.build_block(ii))
        sequence = nn.Sequential(*sequence)
        return sequence

    def forward(self, x):
        x = self.first_operator(x)
        x = einops.rearrange(x,
                             "N (C Y X) -> N C Y X",
                             C=self.first_image_shape[0],
                             Y=self.first_image_shape[1],
                             X=self.first_image_shape[2])
        x = self.network_operator_sequence(x)
        return x


class MSDAE(nn.Module):
    def __init__(self,
                 input_shape,
                 depth,
                 base_channels,
                 growth_rate,
                 latent_dimension,
                 max_dilation,
                 MSD_depth,
                 final_action=None):
        super().__init__()

        self.input_shape = input_shape
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.latent_dimension = latent_dimension
        self.max_dilation = max_dilation
        self.MSD_depth = MSD_depth
        self.final_action = final_action
        if final_action == "Sigmoid":
            self.final_action = nn.Sigmoid()

        self.encoder = MSDEncoder(input_shape=self.input_shape,
                                  depth=self.depth,
                                  base_channels=self.base_channels,
                                  growth_rate=self.growth_rate,
                                  latent_dimension=self.latent_dimension,
                                  max_dilation=self.max_dilation,
                                  MSD_depth=self.MSD_depth)

        self.decoder = MSDDecoder(output_shape=self.input_shape,
                                  depth=self.depth,
                                  base_channels=self.base_channels,
                                  growth_rate=self.growth_rate,
                                  latent_dimension=self.latent_dimension,
                                  max_dilation=self.max_dilation,
                                  MSD_depth=self.MSD_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.final_action is not None:
            x = self.final_action(x)
        return x
