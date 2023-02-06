#!/bin/bash
#
#
# This script runs papermill on all notebooks listed.
# As command line argument, please pass in the name of the jupyter
# kernel needed to execute the code.
#
# See:
# https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084
#
# on how to add your conda environment to your jupyter notebook
# don't forget to make this script executable.
#
# chmod +x run_all.sh
# ./run_all.sh my_kernel
#
# This will take some time to run all. You can inspect all results with jupyter
#
papermill denoising_MSDNet_SMSNetEnsemble.ipynb denoising_MSDNet_SMSNetEnsemble.ipynb -k $1
papermill denoising_selfSupervised.ipynb denoising_selfSupervised.ipynb -k $1
papermill ensembleLearning_SMSNets.ipynb ensembleLearning_SMSNets.ipynb -k $1
papermill imageClassification_SMSNetAutoencoderEnsemble.ipynb imageClassification_SMSNetAutoencoderEnsemble.ipynb -k $1
papermill imageClassification_SMSNetEnsemble.ipynb imageClassification_SMSNetEnsemble.ipynb -k $1
papermill latentSpaceExploration_SMSNetAutoencoders.ipynb latentSpaceExploration_SMSNetAutoencoders.ipynb -k $1
papermill saving_loading_networks.ipynb saving_loading_networks.ipynb -k $1
papermill segmentation_MSDNet_TUNet_TUNet3plus.ipynb segmentation_MSDNet_TUNet_TUNet3plus.ipynb -k $1


