# RNA-seq-denoising

This repository contains code inspired by the paper "Single-cell RNA-seq denoising using a deep count autoencoder" by Eraslan et al. (2019). The code is written in Python and uses the PyTorch library. It is designed to denoise single-cell RNA-seq data using a deep count autoencoder.

## Overview
The folder `src` contains the implementation of the deep count autoencoder along with data loading and loss functions. The module `autoencoder.py` contains an autoencoder base class and implementations for a Poisson autoencoder a negative binomial autoencoder and a zero-inflated negative binomial autoencoder. The module `loss.py` contains the loss functions for the autoencoders. The module `data.py` contains the data loading and preprocessing functions.

The script `train_model.py` contains the training loop. On the top you can specify some parameters, such as the `MODEL_TYPE` that can be either `poisson`, `nb` or `zinb`, the number of epochs, the batch size, the learning rate, the layers, the dropout rate, the device, ...
Run the script with `python train_model.py`.