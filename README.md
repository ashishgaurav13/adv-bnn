# Leveraging Bayesian Neural Networks for Adversarial Attack and Defense Through Data Augmentation 


## Pre-trained Models

Sampled models from a pretrained Bayesian Neural network can be found [here](https://drive.google.com/drive/folders/1A4OriEe-hTCQY3q02FT7wFXJt0BXNZsi?usp=sharing).

The folder contains 3 files 

* lenet-mnist.pt
* alexnet-cifar10.pt
* alexnet-cifar100.pt

Each file contains a list of 100 PyTorch *state_dict* objects that were created by sampling the Bayesian Neural Network. The *lenet-mnist.pt* models were sampled from a Bayesian Neural Network trained on the MNIST dataset using the LeNet architecture for 20 epochs. The *alexnet-cifar\*.pt* models were sampled from a Bayesian Neural Network trained on the CIFAR10 or CIFAR100 dataset using the AlexNet architecture for 200 epochs.

These files can be used by placing them in the top-level directory of the repository. Doing so will bypass training procedure in the *advbnn-cnn\*.ipynb* notebooks.