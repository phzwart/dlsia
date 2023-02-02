import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from dlsia.core.networks import scale_up_down
from torch.autograd import Variable


def max_pool_size_result(Nin, kernel, stride, dilation=1, padding=0):
    """
    Determine the spatial dimension size after a max pooling operation

    :param Nin: dimension of 1d array
    :param kernel: kernel size
    :param stride: stride; might need to match kernel size
    :param dilation: dilation factor

    :param padding: padding parameter
    :return: the resulting array length
    """
    Nout = ((Nin + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
    Nout = int(Nout)
    return Nout


def unet_sizing_chart(N, depth, stride, maxpool_kernel_size,
                      up_down_padding=0, dilation=1):
    """
    Build a set of dictionaries that are useful to make sure that we can map
    arrays back to the right sizes for each downsampling and upsampling
    operation.

    :param N: dimension of array
    :param depth: the total depth of the unet
    :param stride: the stride - we fix this for a single UNet
    :param maxpool_kernel_size: the max pooling kernel size
    :param up_down_padding: max pooling and convT padding, Default is 0
    :param dilation: the dilation factor. default is 1
    :return: a dictionary with information regarding each upsampling and
             downsampling operation

    The data associated with key "Sizes" provides images size per depth
    The data associated with key "Pool Setting" provides info needed to
    construct a MaxPool operator The data associated with key "convT
    Setting" provides info need to construct transposed convolutions such
    that the image of a the right size is constructed.

    """
    resulting_sizes = {}
    convT_settings = {}
    pool_settings = {}

    Nin = N
    for ii in range(depth):
        resulting_sizes[ii] = {}
        convT_settings[ii + 1] = {}
        pool_settings[ii] = {}

        Nout = max_pool_size_result(Nin,
                                    stride=stride,
                                    kernel=maxpool_kernel_size,
                                    dilation=dilation,
                                    padding=up_down_padding
                                    )
        # padding=(maxpool_kernel_size - 1) / 2

        pool_settings[ii][ii + 1] = {"padding": up_down_padding,
                                     "kernel": maxpool_kernel_size,
                                     "dilation": dilation,
                                     "stride": stride
                                     }

        resulting_sizes[ii][ii + 1] = (Nin, Nout)

        outp = scale_up_down.get_outpadding_convT(Nout, Nin,
                                                  dil=dilation,
                                                  stride=stride,
                                                  ker=maxpool_kernel_size,
                                                  padding=up_down_padding
                                                  )

        Nup = scale_up_down.resulting_convT_size(Nout,
                                                 dil=dilation,
                                                 pad=up_down_padding,
                                                 stride=stride,
                                                 ker=maxpool_kernel_size,
                                                 outp=outp
                                                 )

        assert (Nin == Nup)

        convT_settings[ii + 1][ii] = {"padding": up_down_padding,
                                      "output_padding": outp,
                                      "kernel": maxpool_kernel_size,
                                      "dilation": dilation,
                                      "stride": stride
                                      }

        Nin = Nout

    results = {"Sizes": resulting_sizes,
               "Pool_Settings": pool_settings,
               "convT_settings": convT_settings}
    return results


def skip_connection_sizing_chart(dim, chart, kernel_size, stride,
                                 encoder_channels, decoder_channels,
                                 carryover_channels,
                                 up_down_padding=0, dilation=1):
    """
    Build a sizing chart for the (depth * (depth - 1)) skip connections,
    including the required contraction/expansion/concatenations operators and
    corresponding operator parameters. In total, each decoder will
    receive a single connection from each layer.

    :param dim: x or y (or z) spatial dimension index
    :param chart: the skip connection sizing chart; for extracting the outgoing
                  and incoming spatial dimensions at each layer.
    :param kernel_size: The size of the upsampling and downsampling kernel used
                        to step up and down.
    :param stride: The stride we want to use. Typically defaulted to the
                   kernel_size
    :param encoder_channels: TBD
    :param decoder_channels: TBD
    :param carryover_channels: TBD
    :param up_down_padding: upsampling and downsampling padding, Default is 0
    :param dilation: the dilation factor. Default is 1

    :return: A dictionary with information regarding the appropriate up/down
             operations or concatenation and required parameters.
    """

    size_guide = chart[dim]['Sizes']
    depth = len(size_guide)

    # print('')
    # print('Size Guide: ', size_guide)
    # print('')

    # Get dimensions at each layer calculated in unet_sizing_chart
    sizes = []
    for j in range(depth):
        sizes.append(size_guide[j][j + 1][0])

    # Preallocate the operations and necessary parameters
    skip_connections = {}

    # Loop through each incoming encoder/decoder layer
    for in_layer in range(depth):

        # Preallocate the connection
        skip_connections[in_layer] = {}

        # Loop through each terminal decoder layer
        for out_layer in range(depth - 1):

            # Log incoming and outgoing spatial dimensions
            Nin = sizes[in_layer]
            Nout = sizes[out_layer]

            # Factor tells us which power to raise the kernel size and stride
            factor = np.abs(out_layer - in_layer)
            if in_layer == out_layer:
                operation = 'cat'
                padding = None
                outp = None
                channels = encoder_channels[in_layer]
            elif in_layer < out_layer:
                operation = 'downsample'
                outp = None
                channels = encoder_channels[in_layer]

                # Padding may be used to alter outgoing shape, though it's
                # typically not needed; maxpooling defaults to  ceil_mode=False
                padding = 0
            else:
                operation = 'upsample'
                padding = 0

                if in_layer == (depth - 1):
                    channels = encoder_channels[in_layer]

                else:
                    channels = carryover_channels * depth

                # Outpadding required when upsampling through multiple layers
                outp = scale_up_down.get_outpadding_convT(Nin, Nout,
                                                          dil=dilation,
                                                          stride=stride ** factor,
                                                          ker=kernel_size ** factor,
                                                          padding=up_down_padding
                                                          )

            skip_connections[in_layer][out_layer] = {"operation": operation,
                                                     'size_in': Nin,
                                                     'size_out': Nout,
                                                     'channels': channels,
                                                     'factor': factor,
                                                     'kernel': kernel_size,
                                                     'stride': kernel_size,
                                                     'padding': padding,
                                                     'output_padding': outp}

    return skip_connections


def build_up_operator(chart, from_depth, to_depth, in_channels,
                      out_channels, conv_kernel, key="convT_settings"):
    """
    Build an up sampling operator

    :param chart: An array of sizing charts (one for each dimension)
    :param from_depth: The sizing is done at this depth
    :param to_depth: and goes to this depth
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param conv_kernel: the convolutional kernel we want to use
    :param key: a key we can use - default is fine
    :return: returns an operator
    """
    stride = []
    dilation = []
    kernel = []
    padding = []

    output_padding = []

    for ii in range(len(chart)):
        tmp = chart[ii][key][from_depth][to_depth]
        stride.append(tmp["stride"])
        dilation.append(tmp["dilation"])
        kernel.append(tmp["kernel"])
        padding.append(tmp["padding"])
        output_padding.append(chart[ii][key][from_depth][to_depth]["output_padding"])

    return conv_kernel(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel,
                       stride=stride,
                       padding=padding,
                       output_padding=output_padding)


def build_down_operator(chart, from_depth, to_depth, maxpool_kernel, key="Pool_Settings"):
    """
    Build a down sampling operator

    :param chart: Array of sizing charts (one for each dimension)
    :param from_depth: we start at this depth
    :param to_depth: and go here
    :param maxpool_kernel: the max pooling kernel we want to use
                                      (MaxPool2D or MaxPool3D)
    :param key: a key we can use - default is fine
    :return: An operator with given specs
    """
    stride = []
    dilation = []
    kernel = []
    padding = []

    for ii in range(len(chart)):
        tmp = chart[ii][key][from_depth][to_depth]
        stride.append(tmp["stride"])
        dilation.append(tmp["dilation"])
        kernel.append(tmp["kernel"])
        padding.append(tmp["padding"])

    return maxpool_kernel(kernel_size=kernel,
                          stride=stride,
                          padding=padding)


class TUNet3Plus(nn.Module):
    """
    This function creates a U-Net3+ model, a new variant of the classic
    U-Net mode commonly used for image semantic segmentation. While the model
    takes in an input image and outputs a class-segmented image, this new 3+
    variant supplants the classic U-net layer-to-layer skip connections
    with dense skip connections aggregating features from all network layers,
    thus encouraging more multi-scale learning of features

    In this dlsia implementation, a number of architecture-governing
    hyperparameters may be tuned by the user, including the network depth,
    the number of convolutional channels carried over in each individual
    skip connection, convolutional channel growth rate both within & between
    layers, and the normalization & activation operations following each
    convolution.

    :param image_shape: Image shape we use
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param depth: The total depth, or the number of layers
    :param base_channels: Number of image channels in the first operator;
                          i.e. in_channels->base_channels.

    :param growth_rate: The growth rate of number of channels per depth layer
    :param hidden_rate: How many 'in between' channels do we want? This is
                        relative to the feature channels at a given depth
    :param carryover_channels: Number of channels carried over in each skip
                               connection. Each decoder layer will have
                               (depth * carryover_channels) number of channels.
                               If None, defaults to base_channels.
    :param conv_kernel: The convolution kernel we want to us. Conv2D or Conv3D
    :param kernel_down: How do we steps down? MaxPool2D or MaxPool3D
    :param kernel_up: How do we step up? nn.ConvTranspose2d or
                      nn.ConvTranspose3d
    :param normalization: The normalization operation applied after each
                          convolution
    :param activation: The activation function applied in tandem with the
                       normalization operator
    :param conv_kernel_size: The size of the convolutional kernel we use
    :param maxpool_kernel_size: The size of the max pooling kernel we use to
                                step down
    :param stride: The stride we want to use.
    :param dilation: The dilation we want to use.

    """

    def __init__(self,
                 image_shape,
                 in_channels,
                 out_channels,
                 depth,
                 base_channels,
                 growth_rate=2,
                 hidden_rate=1,
                 carryover_channels=0,
                 conv_kernel=nn.Conv2d,
                 kernel_down=nn.MaxPool2d,
                 kernel_up=nn.Upsample,
                 normalization=nn.BatchNorm2d,
                 activation=nn.ReLU(),
                 conv_kernel_size=3,
                 maxpool_kernel_size=2,
                 stride=2,
                 dilation=1
                 ):
        """
        Construct a tuneable UNet

        :param image_shape: Image shape we use
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param depth: The total depth, or the number of layers
        :param base_channels: Number of image channels in the first operator;
                              i.e. in_channels->base_channels.
        :param growth_rate: The growth rate of the number of channels per
                            depth layer.
        :param hidden_rate: How many 'in between' channels do we want? This is
                            relative to the feature channels at a given depth
        :param carryover_channels: Number of channels carried over in each skip
                                   connection. Each decoder layer will have
                                   (depth * carryover_channels) number of
                                   channels. If None, defaults to base_channels.
        :param conv_kernel: instance of PyTorch convolution class. Accepted are
                            nn.Conv1d, nn.Conv2d, and nn.Conv3d.
        :param kernel_down: How do we steps down? MaxPool2D or MaxPool3D
        :param kernel_up: How do we step up? nn.ConvTranspose2d ore
                          nn.ConvTranspose3d
        :param normalization: PyTorch normalization class applied to each
                              layer. Passed as class without parentheses since
                              we need a different instance per layer.
                              ex) normalization=nn.BatchNorm2d
        :param activation: torch.nn class instance or list of torch.nn class
                           instances
        :param conv_kernel_size: The size of the convolutional kernel we use
        :param maxpool_kernel_size: The size of the max pooling/transposed
                                    convolutional kernel we use in
                                    encoder/decoder paths. Default is 2.
        :param stride: The stride we want to use. Controls contraction/growth
                       rates of spatial dimensions (x and y) in encoder/decoder
                       paths. Default is 2.
        :param dilation: The dilation we want to use.
        """
        super().__init__()
        # define the front and back of our network
        self.stride = stride
        self.image_shape = image_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        # determine the overall architecture
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.hidden_rate = hidden_rate

        # These are the convolution / pooling kernels
        self.conv_kernel = conv_kernel
        self.kernel_down = kernel_down
        self.kernel_up = kernel_up

        # print('')
        # print('*** Early Check self.kernel_up: ')
        # print(self.kernel_up.__class__.__name__)
        # print(self.kernel_up.__class__)
        # print('')

        # These are the convolution / pooling kernel sizes
        self.conv_kernel_size = conv_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size

        # These are the number of channels carried over in each skip
        # connection. Defaults to the u

        if carryover_channels == 0:
            self.carryover_channels = base_channels
        else:
            self.carryover_channels = carryover_channels

        # These control the contraction/growth rates of the spatial dimensions
        self.stride = maxpool_kernel_size
        self.dilation = dilation

        # normalization and activation functions
        if normalization is not None:
            self.normalization = normalization
        else:
            self.normalization = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = None
        self.return_final_layer_ = False

        # we now need to get the sizing charts sorted
        self.sizing_chart = []
        for N in self.image_shape:
            self.sizing_chart.append(unet_sizing_chart(N=N,
                                                       depth=self.depth,
                                                       stride=self.stride,
                                                       maxpool_kernel_size=self.maxpool_kernel_size,
                                                       dilation=self.dilation))

        # setup the layers and partial / outputs
        self.encoder_layer_channels_in = {}
        self.encoder_layer_channels_out = {}
        self.encoder_layer_channels_middle = {}

        self.decoder_layer_channels_in = {}
        self.decoder_layer_channels_out = {}
        self.decoder_layer_channels_middle = {}

        self.partials_encoder = {}
        self.partials_decoder = {}

        self.encoders = {}
        self.decoders = {}
        self.step_down = {}
        # self.step_up = {}

        # first pass
        self.encoder_layer_channels_in[0] = self.in_channels
        self.decoder_layer_channels_out[0] = self.base_channels

        for ii in range(self.depth):

            # Match interlayer channels for stepping down
            if ii > 0:
                self.encoder_layer_channels_in[ii] = self.encoder_layer_channels_out[ii - 1]
            else:
                self.encoder_layer_channels_middle[ii] = int(self.base_channels)

            # Set base channels in first layer
            if ii == 0:
                self.encoder_layer_channels_middle[ii] = int(self.base_channels)
            else:
                self.encoder_layer_channels_middle[ii] = int(self.encoder_layer_channels_in[ii] * self.growth_rate)

            # Apply hidden rate for growth within layers
            self.encoder_layer_channels_out[ii] = int(self.encoder_layer_channels_middle[ii] * self.hidden_rate)

            # Decoder layers match Encoder channels

            # Decoders in UNet 3+ receive channel-concatenated layers
            # Question remains: do we want a growth factor for middle channel
            self.decoder_layer_channels_in[ii] = self.carryover_channels * depth
            self.decoder_layer_channels_middle[ii] = self.carryover_channels
            self.decoder_layer_channels_out[ii] = self.carryover_channels

            self.partials_encoder[ii] = None
            self.partials_decoder[ii] = None

        # Bottom encoder layer must adhere to carry_over channel output
        # self.encoder_layer_channels_out[ii] = self.carryover_channels

        for ii in range(depth):

            if ii < (depth - 1):

                # Build encoder/decoder layers
                self.encoders[ii] = "Encode_%i" % ii
                tmp = self.build_unet_layer(self.encoder_layer_channels_in[ii],
                                            self.encoder_layer_channels_middle[ii],
                                            self.encoder_layer_channels_out[ii])
                self.add_module(self.encoders[ii], tmp)

                self.decoders[ii] = "Decode_%i" % ii

                if ii == 0:
                    tmp = self.build_output_layer(
                        self.decoder_layer_channels_in[ii],
                        self.decoder_layer_channels_middle[ii],
                        self.decoder_layer_channels_out[ii],
                        self.out_channels)
                    self.add_module(self.decoders[ii], tmp)
                else:
                    tmp = self.build_unet_layer(self.decoder_layer_channels_in[ii],
                                                self.decoder_layer_channels_middle[ii],
                                                self.decoder_layer_channels_out[ii])
                    self.add_module(self.decoders[ii], tmp)
            else:
                self.encoders[ii] = "Final_layer_%i" % ii
                tmp = self.build_unet_layer(self.encoder_layer_channels_in[ii],
                                            self.encoder_layer_channels_middle[
                                                ii],
                                            self.encoder_layer_channels_out[
                                                ii])
                self.add_module(self.encoders[ii], tmp)

            # Build stepping operations
            if ii < self.depth - 1:
                # we step down like this
                self.step_down[ii] = "Step Down %i" % ii
                tmp = build_down_operator(chart=self.sizing_chart,
                                          from_depth=ii,
                                          to_depth=ii + 1,
                                          maxpool_kernel=self.kernel_down,
                                          key="Pool_Settings")
                self.add_module(self.step_down[ii], tmp)

            # Delete
            # if (ii >= 0) and (ii < depth - 1):
            #    # we step up like this

            #    self.step_up[ii] = "Step Up %i" % ii
            #    if ii == (depth - 2):
            #        tmp = build_up_operator(chart=self.sizing_chart,
            #                                from_depth=ii + 1,
            #                                to_depth=ii,
            #                                in_channels=self.encoder_layer_channels_out[ii + 1],
            #                                out_channels=self.encoder_layer_channels_out[ii],
            #                                conv_kernel=self.kernel_up,
            #                                key="convT_settings")
            #    else:
            #        tmp = build_up_operator(chart=self.sizing_chart,
            #                                from_depth=ii + 1,
            #                                to_depth=ii,
            #                                in_channels=self.decoder_layer_channels_out[ii + 1],
            #                                out_channels=self.encoder_layer_channels_out[ii],
            #                                conv_kernel=self.kernel_up,
            #                                key="convT_settings")

            #    self.add_module(self.step_up[ii], tmp)

        # we now need to get the sizing charts sorted
        self.sizing_chart = []
        for N in self.image_shape:
            self.sizing_chart.append(unet_sizing_chart(N=N,
                                                       depth=self.depth,
                                                       stride=self.stride,
                                                       maxpool_kernel_size=self.maxpool_kernel_size,
                                                       dilation=self.dilation))

        # Now we get all the skip connection information
        self.skip_connection_chart = []
        for N in range(len(self.image_shape)):
            self.skip_connection_chart.append(skip_connection_sizing_chart(N,
                                                                           self.sizing_chart,
                                                                           self.maxpool_kernel_size,
                                                                           self.stride,
                                                                           self.encoder_layer_channels_out,
                                                                           self.decoder_layer_channels_out,
                                                                           self.carryover_channels
                                                                           )
                                              )

        self.skip_connections = [[0 for i in range(depth - 1)] for j in range(depth)]

        # Loop through each incoming encoder/decoder layer
        for in_layer in range(depth):
            # Loop through each terminal decoder layer
            for out_layer in range(depth - 1):
                tmp = self.build_skip_connections(self.skip_connection_chart,
                                                  from_depth=in_layer,
                                                  to_depth=out_layer,
                                                  squeeze_channels=self.carryover_channels
                                                  )

                self.skip_connections[in_layer][out_layer] = f"Skip_connection_{in_layer}_to_{out_layer}"

                self.add_module(self.skip_connections[in_layer][out_layer], tmp)

    def build_skip_connections(self, chart,
                               from_depth,
                               to_depth,
                               squeeze_channels
                               ):

        """
        Single skip connections are built here. Each skip connection consists
        of either: 1) concatenation if incoming layer matches terminal layer,
        2) downsampling operation (MaxPool default) if incoming layer is higher
        than terminal layer, and 3) upsampling operation (ConvTranspose
        default) if incoming layer is lower thatn terminal layer. Subsequently,
        a convolution is applied to squeeze the feature map channels, followed
        by the chosen normalization and activation operations (BatchNorm and
        ReLU by default). All operations applied are defined in TUNet
        __init__, including: self.conv_kernel, self.normalization,
        self.activation, self.kernel_down, and self.kernel_down.


        :param chart: the skip connection sizing chart; contains all
                      information for correctly upsampling and downsampling.
        :param from_depth: The incoming layer size.
        :param to_depth: The terminal layer size.
        :param squeeze_channels: The number of outgoing channels for each
                                 convolution application
        """
        operation = chart[0][from_depth][to_depth]['operation']

        in_channels = []
        # out_channels = []
        size_in = []
        size_out = []
        factor = []
        kernel = []
        stride = []
        padding = []
        output_padding = []

        for ii in range(len(chart)):
            tmp = chart[ii][from_depth][to_depth]

            in_channels.append(tmp['channels'])
            # out_channels.append(['out_channels'])
            size_in.append(tmp['size_in'])
            size_out.append(tmp['size_out'])
            factor.append(tmp['factor'])
            kernel.append(tmp["kernel"])
            stride.append(tmp["stride"])
            padding.append(tmp["padding"])
            output_padding.append(tmp["output_padding"])

        # Preallocate modules to house each skip connection modules
        modules = []

        # Apply upsampling operation
        if operation == 'upsample':

            # Check if transposed convolution
            if self.kernel_up.__name__ == 'ConvTranspose2d':
                # Preserve tuple type after applying factor
                new_kernel = tuple(kernel ** factor[0])
                new_stride = tuple(kernel ** factor[0])

                modules.append(self.kernel_up(in_channels=in_channels[0],
                                              out_channels=in_channels[0],
                                              kernel_size=new_kernel,
                                              stride=new_stride,
                                              output_padding=output_padding
                                              )
                               )
            # Check if upsampling
            elif self.kernel_up.__name__ == 'Upsample':
                # Preserve tuple type after applying factor
                new_size = size_out
                modules.append(self.kernel_up(new_size,
                                              mode='nearest'
                                              )
                               )

        # Apply downsampling operation
        elif operation == 'downsample':

            # Preserve tuple type after applying factor
            new_kernel = tuple(kernel ** factor[0])
            new_stride = tuple(kernel ** factor[0])

            modules.append(self.kernel_down(kernel_size=new_kernel,
                                            stride=new_stride,
                                            padding=padding
                                            )
                           )

        # Apply convolution and normalization/activation bundle to all
        modules.append(self.conv_kernel(in_channels=in_channels[0],
                                        out_channels=squeeze_channels,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )
        if self.normalization is not None:
            modules.append(self.normalization(squeeze_channels))
        if self.activation is not None:
            modules.append(self.activation)

        # Finally, put all modules together in nn.Sequential
        operator = nn.Sequential(*modules)

        return operator

    def build_unet_layer(self, in_channels, in_between_channels, out_channels):
        """
        Build a sequence of convolutions with activations functions and
        normalization layers

        :param in_channels: input channels
        :param in_between_channels: the in between channels
        :param out_channels: the output channels
        :return:
        """

        # Preallocate modules to house each skip connection modules
        modules = []

        # Add first convolution
        modules.append(self.conv_kernel(in_channels,
                                        in_between_channels,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(in_between_channels))
        if self.activation is not None:
            modules.append(self.activation)

        # Add second convolution
        modules.append(self.conv_kernel(in_between_channels,
                                        out_channels,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(out_channels))
        if self.activation is not None:
            modules.append(self.activation)

        # Finally, wrap all modules together in nn.Sequential
        operator = nn.Sequential(*modules)

        return operator

    def build_output_layer(self, in_channels,
                           in_between_channels1,
                           in_between_channels2,
                           final_channels):
        """
        For final output layer, builds a sequence of convolutions with
        activations functions and normalization layers

        :param final_channels: The output channels
        :type final_channels: int
        :param in_channels: input channels
        :param in_between_channels1: the in between channels after first convolution
        :param in_between_channels2: the in between channels after second convolution
        "param final_channels: number of channels the network outputs
        :return:
        """

        # Preallocate modules to house each skip connection modules
        modules = []

        # Add first convolution
        modules.append(self.conv_kernel(in_channels,
                                        in_between_channels1,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(in_between_channels1))
        if self.activation is not None:
            modules.append(self.activation)

        # Add second convolution
        modules.append(self.conv_kernel(in_between_channels1,
                                        in_between_channels2,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(in_between_channels2))
        if self.activation is not None:
            modules.append(self.activation)

        # Append final output convolution
        modules.append(self.conv_kernel(in_between_channels2,
                                        final_channels,
                                        kernel_size=1
                                        )
                       )

        # Finally, wrap all modules together in nn.Sequential
        operator = nn.Sequential(*modules)

        return operator

    def forward(self, x):
        """
        Default forward operator.

        :param x: input tensor.
        :return: output of neural network
        """

        # first pass through the encoder
        for ii in range(self.depth - 1):
            # channel magic
            x_out = self._modules[self.encoders[ii]](x)

            # store this for decoder side processing
            self.partials_encoder[ii] = x_out

            # step down
            x = self._modules[self.step_down[ii]](x_out)
            # done

            # print('')
            # print('After stepping down from layer: ', ii)
            # print('Number of params: ', helpers.count_parameters(self._modules[self.encoders[ii]]))
            # print(self._modules[self.encoders[ii]])
            # print('Number of params: ', helpers.count_parameters(self._modules[self.step_down[ii]]))
            # print(self._modules[self.step_down[ii]])
            # print(x.size())
            # print('')

        # last convolution in bottom, no need to stash results
        x_out = self._modules[self.encoders[self.depth - 1]](x)
        self.partials_decoder[self.depth - 1] = x_out

        # print('')
        # print('After bottom layer: ', self.depth - 1)
        # print(self._modules[self.encoders[self.depth - 1]])
        # print('Number of params: ', helpers.count_parameters(self._modules[self.encoders[self.depth - 1]]))
        # print(x_out.size())
        # print('')

        # print('Encoder 0 out: ', self.partials_encoder[0].size())
        # print('Encoder 1 out: ', self.partials_encoder[1].size())
        # print('Encoder 2 out: ', self.partials_encoder[2].size())
        # print('Encoder 3 out: ', self.partials_encoder[3].size())
        # print('Decoder 4 out: ', self.partials_decoder[4].size())
        # print('')

        for out_layer in range(self.depth - 2, -1, -1):

            tmp = []
            for in_layer in range(self.depth):

                # print('In layer to out layer: ', in_layer, out_layer) print('Number of params: ',
                # helpers.count_parameters(self._modules[self.skip_connections[in_layer][out_layer]])) print(
                # self._modules[self.skip_connections[in_layer][out_layer]])

                if in_layer <= out_layer:
                    xin = self.partials_encoder[in_layer]

                    # print('Used Encoder: ', in_layer)

                elif in_layer == (out_layer + 1):
                    xin = self.partials_decoder[in_layer]
                    # xin = x_out
                    # print('Used previous output in layer: ', in_layer)
                    # print('Shape of Xin: ', xin.size())

                else:
                    xin = self.partials_decoder[in_layer]

                    # print('Used Decoder: ', in_layer)

                # print('Xin: ', xin.size())
                tmp.append(self._modules[self.skip_connections[in_layer][out_layer]](xin))

                # print('Xout: ', tmp[in_layer].size())

                # print('')
            xout = torch.cat(tmp, dim=1)
            self.partials_decoder[out_layer] = xout
            # print('New size after all concatenations: ', xout.size())
            # print('Decoder at layer: ', out_layer, self.partials_decoder[out_layer].size())

        x_out = self._modules[self.decoders[0]](xout)

        return x_out

    def topology_dict(self):
        """
        Get all parameters needed to build this network

        :return: An orderdict with all parameters needed
        :rtype: OrderedDict
        """

        topo_dict = OrderedDict()
        topo_dict["image_shape"] = self.image_shape
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_channels"] = self.out_channels
        topo_dict["depth"] = self.depth
        topo_dict["base_channels"] = self.base_channels
        topo_dict["growth_rate"] = self.growth_rate
        topo_dict["hidden_rate"] = self.hidden_rate
        topo_dict["carryover_channels"] = self.carryover_channels
        topo_dict["conv_kernel"] = self.conv_kernel
        topo_dict["kernel_down"] = self.kernel_down
        topo_dict["kernel_up"] = self.kernel_up
        topo_dict["normalization"] = self.normalization
        topo_dict["activation"] = self.activation
        topo_dict["conv_kernel_size"] = self.conv_kernel_size
        topo_dict["maxpool_kernel_size"] = self.maxpool_kernel_size
        topo_dict["stride"] = self.stride
        topo_dict["dilation"] = self.dilation
        return topo_dict

    def save_network_parameters(self, name=None):
        """
        Save the network parameters
        :param name: The filename
        :type name: str
        :return: None
        :rtype: None
        """
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)


def TUNetwork3Plus_from_file(filename):
    """
    Construct an MSDNet from a file with network parameters

    :param filename: the filename
    :type filename: str
    :return: An SMSNet
    :rtype: SMSNet
    """
    network_dict = torch.load(filename, map_location=torch.device('cpu'))
    TUNet3PlusObj = TUNet3Plus(**network_dict["topo_dict"])
    TUNet3PlusObj.load_state_dict(network_dict["state_dict"])
    return TUNet3PlusObj


def tst():
    a = 128
    b = 111
    obj = TUNet3Plus(image_shape=(a, b),
                     in_channels=1,
                     out_channels=1,
                     depth=5,
                     base_channels=16,
                     growth_rate=2,
                     hidden_rate=1,
                     maxpool_kernel_size=2,
                     normalization=None,
                     activation=None
                     )

    print(obj)

    from torchsummary import summary
    from pyMSDtorch.core import helpers
    device = helpers.get_device()
    obj.to(device)
    summary(obj, (1, a, b))

    x = Variable(
        torch.rand(3, 1, a, b))
    x = x.cuda()

    from time import time
    starttime = time()

    times = []
    for j in range(50):
        start = time()

        x = obj(x)
        torch.cuda.empty_cache()

        end = time()
        elapsed = end - start
        times.append(elapsed)
    endtime = time()

    print('First run time: ', times[0])

    print('Avg of all others: ', sum(times[1:]) / (len(times) - 1))

    # print('Input shape: ', x.size())
    # print('Output size: ', y.size())
    return True


if __name__ == "__main__":
    assert tst()
