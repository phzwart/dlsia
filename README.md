![logo](docs/images/dlsia.png 'the logo')


# Welcome to dlsia's documentation!

<a style="text-decoration:none !important;" href="https://dlsia.readthedocs.io/en/latest/" alt="website"><img src="https://img.shields.io/readthedocs/dlsia" /></a>
<a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
<a style="text-decoration:none !important;" href="https://img.shields.io/github/commit-activity/m/phzwart/dlsia" alt="License"><img src="https://img.shields.io/github/commit-activity/m/phzwart/dlsia" /></a>
![GitHub contributors](https://img.shields.io/github/contributors/phzwart/dlsia)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/phzwart/dlsia)

dlsia (Deep Learning for Scientific Image Analysis) provides easy access to a number of segmentation and denoising
methods using convolution neural networks. The tools available are build for 
microscopy and synchrotron-imaging/scattering data in mind, but can be used 
elsewhere as well.

The easiest way to start playing with the code is to install dlsia and 
perform denoising/segmenting using custom neural networks in our tutorial 
notebooks located in the dlsia/tutorials folder.

## Install dlsia

We offer several methods for installation. 

### pip: Python package installer

We are currently working on a stable release.

### From source

dlsia may be directly downloaded and installed into your machine by 
cloning the public repository into an empty directory using:

```console
$ git clone https://github.com/phzwart/dlsia.git
```

Once cloned, move to the newly minted dlsia directory and install 
dlsia using:

```console
$ cd dlsia
$ pip install -e .
```

### Further documentation & tutorial download

For more in-depth documentation and end-to-end training workflows, please 
visit our 
[readthedocs](https://dlsia.readthedocs.io/en/latest/index.html) page 
for more support. To download only the tutorials in a new folder, use the 
following terminal input for a sparse git checkout:

```console
mkdir dlsiaTutorials
cd dlsiaTutorials
git init
git config core.sparseCheckout true
git remote add -f origin https://github.com/phzwart/dlsia.git
echo "dlsia/tutorials/*" > .git/info/sparse-checkout
git checkout main
```

## Getting started

We start with some basic imports - we import a network and some training 
scripts:

```python
from dlsia.core.networks import msdnet
from dlsia.core import train_scripts
```

### Mixed-Scale dense networks (MSDNet)

![msdnet](docs/images/MSDNet_fig.png 'msdnet fig')


A plain 2d mixed-scale dense network is constructed as follows:

```python
from dlsia.core.networks import msdnet

msdnet_model = msdnet.MixedScaleDenseNetwork(in_channels=1,
                                             out_channels=1,
                                             num_layers=20,
                                             max_dilation=10)
```

while 3d network types for volumetric images can be built passing in equivalent 
kernels for 3 dimensions:

```python
import torch
from torch import nn

msdnet3d_model = msdnet.MixedScaleDenseNetwork(in_channels=1,
                                               out_channels=1,
                                               num_layers=20,
                                               max_dilation=10,
                                               normalization=nn.BatchNorm3d,
                                               convolution=nn.Conv3d)
```

Note that each instance of a convolution operator is followed by ReLU 
activation and batch normalization. To turn these off, simply pass in the 
parameters

```python
activation=None,
normalization=None
```

### Sparse mixed-scale dense network (SMSNet)

![smsnet](docs/images/RMSNet_fig.png 'smsnet fig')


The dlsia suite also provides ways and means to build random, sparse mixed 
scale networks. SMSNets contain more sparsely connected nodes than a standard 
MSDNet and are useful to alleviate overfitting and multi-network aggregation. 
Controlling sparsity is possible, see full documentation for more details.

```python
from dlsia.core.networks import smsnet

smsnet_model = smsnet.random_SMS_network(in_channels=1,
                                         out_channels=1,
                                         layers=20,
                                         dilation_choices=[1, 2, 4, 8],
                                         hidden_out_channels=[1, 2, 3])
```
### Tunable U-Nets

![tunet](docs/images/UNet_fig.png 'tunet fig')

An alternative network choice is to construct a UNet. Classic U-Nets can easily 
explode in the number of parameters it requires; here we make it a bit easier 
to tune desired architecture-governing parameters:

```python
from dlsia.core.networks import tunet

tunet_model = tunet.TUNet(image_shape=(64, 128),
                          in_channels=1,
                          out_channels=4,
                          base_channels=4,
                          depth=3,
                          growth_rate=1.5)
```

## Training

### Data preparation

To prep data for training, we make liberal use of PyTorch DataLoader 
classes. This allows for easy handling of data in the training process and 
automates the iterative loading of batch sizes.

In the example below, we take pair two numpy arrays of shape ```[num_images, 
num_channels, x_size, y_size]``` consisting of training images and masks, convert 
them into PyTorch tensors, then initialize the DataLoader class.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.Tensor(training_imgs), 
                           torch.Tensor(training_masks))

train_loader_params = {'batch_size': 20,
                       'shuffle': True}

train_loader = DataLoader(train_data, **train_loader_params)
```

### Training loop

Once your DataLoaders are constructed, the training of these networks is as 
simple as defining a torch.nn optimizer, and calling the training script:

```python
from torch import optim, nn
from dlsia.core import helpers, train_scripts

criterion = nn.CrossEntropyLoss()   # For segmenting
optimizer = optim.Adam(tunet_model.parameters(), lr=1e-2)

device = helpers.get_device()
tunet_model = tunet_model.to(device)

tunet_model, results = train_scripts.train_segmentation(net=tunet_model,
                                                        trainloader=train_loader,
                                                        validationloader=test_loader,
                                                        NUM_EPOCHS=epochs, 
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        device=device,
                                                        show=1)
```

The output of the training scripts is the trained network and a dictionary with 
training losses and evaluation metrics. You can view them as follows:

```python
from dlsia.viz_tools import plots

fig = plots.plot_training_results_segmentation(results)
fig.show()
```

## Saving and loading models

Each dlsia network library contains submodules for saving trained 
networks and loading them from file. Using the conventional PyTorch ```.pt ``` 
model file extension, the TUNet above may be saved with

```python
savepath = 'this_tunet.pt'
tunet_model.save_network_parameters(savepath)
```

and reloaded for future use with

```python
copy_of_tunet = tunet.TUNetwork_from_file(savepath)
```

## License and Legal Stuff

This software has been developed from funds that originate from the US tax 
payer and is free for academics. Please have a look at the license agreement 
for more details. Commercial usage will require some extra steps. Please 
contact ipo@lbl.gov for more details.

## Final Thoughts

This documentation is far from complete, but have some notebooks as part of the codebase, which could provide a good
entry point.

More to come!
