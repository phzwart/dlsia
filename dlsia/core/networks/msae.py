"""
Mixed Scale Autoencoders.

This are general objects to construct mixed scale autoencoders.
Substantial testing is needed for their utility.
"""
import einops
import torch
import torch.nn as nn

from dlsia.core.networks import tunet


class single_stack(nn.Module):
    """
    A single stack object, no scale changes.
    in_channels, to out_channels, with given depth and given fixed dilation and kernel size.
    No skip connections.
    """

    def __init__(self, in_channels, out_channels, depth, dilation, conv_kernel_size=3):
        """
        Construct an object that takes an image with in_channels channels and spit an image with out_channels channels.
        A sequence of convolutions with fixed, but specified dilations, followed by ReLU and batchnorm is performed,
        depth times.

        Parameters
        ----------
        in_channels : input channels
        out_channels : output channels
        depth : the depth of network
        dilation : the dilation size
        conv_kernel_size : kernel size
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.ops = []
        dilated_conv_kernel_size = (conv_kernel_size - 1) * dilation + 1
        padding = dilated_conv_kernel_size // 2

        for ii in range(self.depth):
            if ii == 0:
                inc = self.in_channels
            else:
                inc = self.out_channels

            tmp = nn.Conv2d(inc,
                            self.out_channels,
                            conv_kernel_size,
                            dilation=dilation,
                            padding=padding,
                            groups=1,
                            bias=True,
                            padding_mode='zeros')
            self.ops.append(tmp)
            self.ops.append(nn.ReLU())
            self.ops.append(nn.BatchNorm2d(out_channels))
        self.ops = nn.Sequential(*self.ops)

    def forward(self, x):
        """
        A generic forward method
        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output tensor
        """
        x = self.ops(x)
        return x


class MixedScaleWide(nn.Module):
    """
    A parallel set of single_stack objects that are combined in a single output tensor.
    The output values for each pixel of the single stack are combined via a fully connected network.
    """

    def __init__(self, in_channels, out_channels, depth, width, conv_kernel_size=3):
        """
        Construct an object that combines a number single_stack object and aggregate their outputs via a fully connected
        network.

        Parameters
        ----------
        in_channels : input channels
        out_channels : output channels
        depth : the depth
        width : the number of single stacks, with linear incremental dilations
                if width is 4, dilations will 1,2,3,4
        conv_kernel_size : the kernel size
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.width = width
        self.conv_kernel_size = conv_kernel_size
        self.actions = []

        for ii in range(self.width):
            dilation = ii + 1
            ss = single_stack(in_channels, out_channels, depth, dilation)
            self.add_module("Dilation %i" % dilation, ss)
            self.actions.append("Dilation %i" % dilation)

        # TODO add in RELU and batchnorm and added layers
        aggregate = nn.Conv2d(self.out_channels * self.width,
                              self.out_channels,
                              kernel_size=1)

        self.add_module("aggregate", aggregate)

    def forward(self, x):
        """
        Basic forward operator
        Parameters
        ----------
        x : input tensor

        Returns
        -------
        output tensor
        """

        parts = []
        for action in self.actions:
            tmp = self._modules[action](x)
            parts.append(tmp)

        parts = torch.cat(parts, 1)
        x = self._modules["aggregate"](parts)
        return x


class MSEncoder(nn.Module):
    """
    Combine a number of MixedScaleWide objects in sequence with down-sampling operators to do compression.
    This is the encoder part. It leans on some object we build for the TUNet networks.
    """

    def __init__(self,
                 input_shape,
                 depth,
                 base_channels,
                 growth_rate,
                 latent_dimension,
                 max_dilation=5,
                 MS_depth=1):
        """
        Build an object that uses mixed scale networks to compress an image: encoder side

        Parameters
        ----------
        input_shape : input shape
        depth : the depth of the network
        base_channels : the output channels for the mixed scale wide channels
        growth_rate : determines how fast we let the output channels per mixed scale wide object grow over depth
        latent_dimension : the final latent dimension we compress to
        max_dilation : the width (i.e. maximum dilation) of the mixed-scale wide object
        MS_depth : the depth of the mixed scale wide networks
        """

        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.latent_dimension = latent_dimension
        self.max_dilation = max_dilation
        self.MS_depth = MS_depth

        ysizing = tunet.unet_sizing_chart(N=self.input_shape[1], depth=self.depth, stride=2, maxpool_kernel_size=2, dilation=1)
        xsizing = tunet.unet_sizing_chart(N=self.input_shape[2], depth=self.depth, stride=2, maxpool_kernel_size=2, dilation=1)

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
        """
        Build individual blocks.

        Parameters
        ----------
        from_depth : starting at this depth, do the work

        Returns
        -------
        a step in the encoder sequence
        """

        in_channels = self.channel_sequence[from_depth][0]
        out_channels = self.channel_sequence[from_depth][1]

        # a wide mixed scale
        this_wide_net = MixedScaleWide(in_channels=in_channels,
                                       out_channels=out_channels,
                                       depth=self.MS_depth,
                                       width=self.max_dilation)
        # MaxPool
        this_MaxPool = tunet.build_down_operator(chart=self.sizing_chart,
                                                 from_depth=from_depth,
                                                 to_depth=from_depth + 1,
                                                 maxpool_kernel=nn.MaxPool2d,
                                                 key="Pool_Settings")
        operator = nn.Sequential(this_wide_net, this_MaxPool)
        return operator

    def stack_blocks(self):
        """
        Build the blocks and stack them together

        Returns
        -------
        The sequence of encoder operators
        """
        operators = []
        for ii in range(self.depth):
            operators.append(self.build_block(ii))
        operators = nn.Sequential(*operators)
        return operators

    def final_linear_layer(self):
        """
        Build the final linear layer

        Returns
        -------
        the final operator
        """
        in_features = self.final_image_shape[0] * self.final_image_shape[1] * self.final_image_shape[2]
        out_features = self.latent_dimension
        operator = nn.Linear(in_features=in_features, out_features=out_features)
        return operator

    def forward(self, x):
        """
        Standard forward operator

        Parameters
        ----------
        x : inpuit tensor

        Returns
        -------
        output latent vector

        """
        x = self.network_operator_sequence(x)
        x = einops.rearrange(x, "N C Y X -> N (C Y X)")
        x = self.final_operator(x)
        return x


class MSDecoder(nn.Module):
    """
    Inverse of the encoder: latent space to image
    """

    def __init__(self,
                 output_shape,
                 depth,
                 base_channels,
                 growth_rate,
                 latent_dimension,
                 max_dilation=5,
                 MS_depth=1):
        """
        Build an object that uses mixed scale networks to compress an image: decoder side

        Parameters
        ----------
        input_shape : input shape
        depth : the depth of the network
        base_channels : the output channels for the mixed scale wide channels
        growth_rate : determines how fast we let the output channels per mixed scale wide object grow over depth
        latent_dimension : the final latent dimension we compress to
        max_dilation : the width (i.e. maximum dilation) of the mixed-scale wide object
        MS_depth : the depth of the mixed scale wide networks
        """
        super().__init__()
        self.output_shape = output_shape
        self.input_channels = output_shape[0]
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.latent_dimension = latent_dimension
        self.max_dilation = max_dilation
        self.MS_depth = MS_depth

        ysizing = tunet.unet_sizing_chart(N=self.output_shape[1], depth=self.depth, stride=2, maxpool_kernel_size=2, dilation=1)
        xsizing = tunet.unet_sizing_chart(N=self.output_shape[2], depth=self.depth, stride=2, maxpool_kernel_size=2, dilation=1)
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
        """
        First linear layer
        Returns
        -------
        First linear layer
        """
        in_features = self.latent_dimension
        out_features = self.first_image_shape[0] * self.first_image_shape[1] * self.first_image_shape[2]
        operator = nn.Linear(in_features=in_features, out_features=out_features)
        return operator

    def build_block(self, from_depth):
        """
        Build a single block
        Parameters
        ----------
        from_depth : from here

        Returns
        -------
        the operator
        """
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

        # a wide mixed scale
        this_wide_net = MixedScaleWide(in_channels=in_channels,
                                       out_channels=out_channels,
                                       depth=self.MS_depth,
                                       width=self.max_dilation)

        operator = nn.Sequential(this_convT, this_wide_net)
        return operator

    def stack_blocks(self):
        """
        Stack blocks together.
        Returns
        -------
        A sequence of operators
        """
        sequence = []
        for ii in range(0, self.depth)[::-1]:
            sequence.append(self.build_block(ii))
        sequence = nn.Sequential(*sequence)
        return sequence

    def forward(self, x):
        """
        Standard forward operator

        Parameters
        ----------
        x : input latent vector

        Returns
        -------
        output tensor

        """
        x = self.first_operator(x)
        x = einops.rearrange(x,
                             "N (C Y X) -> N C Y X",
                             C=self.first_image_shape[0],
                             Y=self.first_image_shape[1],
                             X=self.first_image_shape[2])
        x = self.network_operator_sequence(x)
        return x


class MSAE(nn.Module):
    """
    An autoencoder.

    """

    def __init__(self,
                 input_shape,
                 depth,
                 base_channels,
                 growth_rate,
                 latent_dimension,
                 max_dilation,
                 MS_depth,
                 final_action=None):
        """
        Build an autoencoder using the building blocks from above.
        Parameters
        ----------
        input_shape : input shape
        depth : the depth of the network
        base_channels : the output channels for the mixed scale wide channels
        growth_rate : determines how fast we let the output channels per mixed scale wide object grow over depth
        latent_dimension : the final latent dimension we compress to
        max_dilation : the width (i.e. maximum dilation) of the mixed-scale wide object
        MS_depth : the depth of the mixed scale wide networks
        final_action : Determines final action. If "Sigmoid", a sigmoid is assigned.
        """
        super().__init__()

        self.input_shape = input_shape
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.latent_dimension = latent_dimension
        self.max_dilation = max_dilation
        self.MS_depth = MS_depth
        self.final_action = final_action
        if final_action == "Sigmoid":
            self.final_action = nn.Sigmoid()

        self.encoder = MSEncoder(input_shape=self.input_shape,
                                 depth=self.depth,
                                 base_channels=self.base_channels,
                                 growth_rate=self.growth_rate,
                                 latent_dimension=self.latent_dimension,
                                 max_dilation=self.max_dilation,
                                 MS_depth=self.MS_depth)

        self.decoder = MSDecoder(output_shape=self.input_shape,
                                 depth=self.depth,
                                 base_channels=self.base_channels,
                                 growth_rate=self.growth_rate,
                                 latent_dimension=self.latent_dimension,
                                 max_dilation=self.max_dilation,
                                 MS_depth=self.MS_depth)

    def forward(self, x):
        """
        Standard forward operator

        Parameters
        ----------
        x : input image / tensor

        Returns
        -------
        output image / tensor
        """
        x = self.encoder(x)
        x = self.decoder(x)
        if self.final_action is not None:
            x = self.final_action(x)
        return x
