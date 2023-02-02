<img src="images/pymsdtorch.png" width=600 />

# Welcome to dlsia's documentation!

<a style="text-decoration:none !important;" href="https://dlsia.readthedocs.io/en/latest/" alt="website"><img src="https://img.shields.io/readthedocs/dlsia" /></a>
<a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>


dlsia (Deep Learning for Scientific Image Analysis) provides easy access to a number of segmentation and denoising
methods using convolution neural networks. The tools available are build for 
microscopy and synchrotron-imaging/scattering data in mind, but can be used 
elsewhere as well.

The easiest way to start playing with the code is to install dlsia and 
perform denoising/segmenting using custom neural networks in our tutorial 
notebooks located in the dlsia/tutorials folder, or perform multi-class 
segmentation in Gaussian noise
on `google colab <https://colab.research.google.
com/drive/1ljMQ12UZ57FJjQ9CqG06PZo-bzOnY-UE?usp=sharing>`

# Install pyMSDtorch

We offer several methods for installation. 

## pip: Python package installer

The latest stable release may be installed with:

```console
$ pip install pymsdtorch .
```

## From source

pyMSDtorch may be directly downloaded and installed into your machine by 
cloning the public repository into an empty directory using:

```console
$ git clone https://bitbucket.org/berkeleylab/pymsdtorch.git .
```

Once cloned, move to the newly minted pymsdtorch directory and install 
pyMSDtorch using:

```console
$ cd pymsdtorch
$ pip install -e .
```

## Tutorials only

To download only the tutorials in a new folder, use the following 
terminal input for a sparse git checkout:

```console
$ mkdir pymsdtorchTutorials
$ cd pymsdtorchTutorials
$ git init
$ git config core.sparseCheckout true
$ git remote add -f origin https://bitbucket.org/berkeleylab/pymsdtorch.git
$ echo "pyMSDtorch/tutorials/*" > .git/info/sparse-checkout
$ git checkout main
```

# Network Initialization

We start with some basic imports - we import a network and some training 
scripts:

```python
from dlsia.core.networks import msdnet
from dlsia.core import train_scripts
```

## Mixed-Scale dense networks (MSDNet)

<img src="images/MSDNet_fig.png" width=600 />


A plain 2d mixed-scale dense network is constructed as follows:

```python
from torch import nn
netMSD2D = MSDNet.MixedScaleDenseNetwork(in_channels=1,
                                        out_channels=1,
                                        num_layers=20,
                                        max_dilation=10,
                                        activation=nn.ReLU(),
                                        normalization=nn.BatchNorm2d,
                                        convolution=nn.Conv2d)

```

while 3D network types for volumetric images can be built passing in equivalent 
kernels:

```python
from torch import nn
netMSD3D = MSDNet.MixedScaleDenseNetwork(in_channels=1,
                                        out_channels=1,
                                        num_layers=20,
                                        max_dilation=10,
                                        activation=nn.ReLU(),
                                        normalization=nn.BatchNorm3d,
                                        convolution=nn.Conv3d)

```

## Sparse mixed-scale dense network (SMSNet)

<img src="images/RMSNet_fig.png" width=600 />


The pyMSDtorch suite also provides ways and means to build random, sparse mixed 
scale networks. SMSNets contain more sparsely connected nodes than a standard 
MSDNet and are useful to alleviate overfitting and multi-network aggregation. 
Controlling sparsity is possible, see full documentation for more details.

```python
from dlsia.core.networks import smsnet

netSMS = smsnet.random_SMS_network(in_channels=1,
                                   out_channels=1,
                                   layers=20,
                                   dilation_choices=[1, 2, 4, 8],
                                   hidden_out_channels=[1, 2, 3])

```
## Tunable U-Nets

<img src="images/UNet_fig.png" width=600 />

An alternative network choice is to construct a UNet. Classic U-Nets can easily 
explode in the number of parameters it requires; here we make it a bit easier 
to tune desired architecture-governing parameters:

```python
from dlsia.core.networks import tunet

netTUNet = tunet.TUNet(image_shape=(121, 189),
                       in_channels=1,
                       out_channels=4,
                       base_channels=4,
                       depth=3,
                       growth_rate=1.5)

```

# Training

If your data loaders are constructed, the training of these networks is as 
simple as defining a torch.nn optimizer, and calling the training script:

```python
from torch import optim, nn
from pyMSDtorch.core import helpers

criterion = nn.CrossEntropyLoss()   # For segmenting
optimizer = optim.Adam(netTUNet.parameters(), lr=1e-2)

device = helpers.get_device()
netTUNet = netTUNet.to(device)
netTUNet, results = train_scripts.train_segmentation(net=netTUNet,
                                           trainloader=train_loader,
                                           validationloader=test_loader,
                                           NUM_EPOCHS=epochs,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           device=device,
                                           show=1)

```

The output of the training scripts is the trained network and a dictionary with 
training losses and evaluationmetrics. You can view them as follows:

```python
from pyMSDtorch.viz_tools import plots
fig = plots.plot_training_results_segmentation(results)
fig.show()

```

# Saving and loading models

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