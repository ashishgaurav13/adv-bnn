# Leveraging Bayesian Neural Networks for Adversarial Attack and Defense


## Pre-trained Models

Sampled models from a pretrained Bayesian Neural network can be found [here](https://drive.google.com/drive/folders/1A4OriEe-hTCQY3q02FT7wFXJt0BXNZsi?usp=sharing).

Each file contains a list of 100 PyTorch *state_dict* objects that were created by sampling the Bayesian Neural Network. The *lenet-mnist.pt* models were sampled from a Bayesian Neural Network trained on the MNIST dataset using the LeNet architecture for 20 epochs. 

These files can be used by placing them in the top-level directory of the repository. Doing so will bypass training procedure in the *advbnn-cnn\*.ipynb* notebooks.

## Experiments and Files

* `advbnn.ipynb` - Introductory example for adversarial attack using BNN and OTCM
* `eps_hyperparam.ipynb` - Epsilon hyperparameter search
* `advbnn_ALL_methods.ipynb` - Implementation of OTCM, FGSM, BIM and ITCM
* `advbnn-cnn.ipynb` - Showcases training a BNN with convolutional architecture, combinations allowed - LeNet/AlexNet with MNIST/CIFAR10/CIFAR100
* `advbnn-2d*.ipynb` - BNN and OTCM on 2D toy dataset
* `generate_adv_examples-OTCM.py` - Generates adversarial OTCM images
* `generate_adv_examples-ITCM.py` - Generates adversarial ITCM images
* `ModelTrainingMNIST.py` - Loads adversarial images and does adversarial training
* `Evaluate-MNIST-OTCM.ipynb` - Evaluates adversarially trained models on different OTCM test sets (Table I)
* `Evaluate-MNIST-ITCM.ipynb` - Evaluates adversarially trained models on different ITCM test sets (Table II)

For adversarial training, please run `generate_adv_examples*.py` to save adversarial examples to disk, followed by `ModelTrainingMNIST.py` to do adversarial training. Before doing adversarial training, ensure that the folder names within `ModelTrainingMNIST.py` are correctly specified. Then, to generate the Table I, II results, run the `Evaluate-MNIST-*.ipynb` files.

## References

* [github.com/kumar-shridhar/PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) - BBB code
* [github.com/paraschopra/bayesian-neural-network-mnist](https://github.com/paraschopra/bayesian-neural-network-mnist) - BNN with Pyro
* [Pyro website](https://pyro.ai/) - Probabilistic programming framework
