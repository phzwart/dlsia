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
papermill tutorial_ensemble_labeling.ipynb tutorial_ensemble_labeling_out.ipynb -k $1
papermill tutorial_AutoEncode_SMS.ipynb tutorial_AutoEncode_SMS_out.ipynb -k $1
papermill tutorial_self_supervised_denoising.ipynb tutorial_self_supervised_denoising_out.ipynb -k $1
papermill tutorial_semantic_segmentation.ipynb tutorial_semantic_segmentation_out.ipynb -k $1
papermill tutorial_randomized_sparse_mixed_scale_networks.ipynb tutorial_randomized_sparse_mixed_scale_networks_out.ipynb -k $1
papermill tutorial_supervised_denoising_2d.ipynb tutorial_supervised_denoising_2d_out.ipynb -k $1
papermill tutorial_segmentation_MSDNet_TUNet_TUNet3plus.ipynb tutorial_segmentation_MSDNet_TUNet_TUNet3plus_out.ipynb -k $1
papermill tutorial_ensemble_averaged_autoencoding_and_labeling.ipynb tutorial_ensemble_averaged_autoencoding_and_labeling_out.ipynb -k $1
papermill tutorial_AutoEncode_and_Label.ipynb tutorial_AutoEncode_and_Label_out.ipynb -k $1
papermill tutorial_saving_and_loading_networks tutorial_saving_and_loading_networks_out.ipynb -k $1

