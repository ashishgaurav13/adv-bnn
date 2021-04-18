import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import pyro
import tqdm
import os
import common
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

import re, pickle
from torch.utils.data import DataLoader, random_split

common.set_seed(1)

# Settings
HOME_DIR = "/home/fcbeylun/adv-bnn/"
HOME_DIR = "./"
EPS  = 0.05
COMB = "champ"

# for GPU
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration parameters
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}
lr_start = 0.001
num_workers = 1
valid_size = 0.2
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 0.1

net_type = 'lenet'   # (lenet/alexnet)
dataset  = 'mnist'    # (mnist/cifar10/cifar100)


if dataset == 'mnist':
    transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
    ])
    inputs = 1
    outputs = 10
    trainset = torchvision.datasets.MNIST(root=HOME_DIR, train=True, download=True, transform=transform)
elif dataset == 'cifar10':
    transform = transforms.Compose([
                transforms.ToTensor(),
    ])
    inputs = 3
    outputs = 10
    trainset = torchvision.datasets.CIFAR10(root=HOME_DIR, train=True, download=True, transform=transform)
elif dataset == 'cifar100':
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
    ])
    inputs = 3
    outputs = 100
    trainset = torchvision.datasets.CIFAR100(root=HOME_DIR, train=True, download=True, transform=transform)
else:
    raise RuntimeException("Unsupported dataset")


if net_type == 'lenet':
    net_class = common.non_bayesian_models.LeNet
    bbb_model = common.bayesian_models.BBBLeNet
elif net_type == 'alexnet':
    net_class = common.non_bayesian_models.AlexNet
    bbb_model = common.bayesian_models.BBBAlexNet
else:
    raise RuntimeException("Unsupported network type")


net = bbb_model(outputs, inputs, priors, layer_type, activation_type).to(device)


def train_and_save_models(epochs = 10, K = 100, modelname = "model-cnn.pt", force_train=False):
    if os.path.exists(modelname) and not force_train:
        print("File exists")
        return
    # Train with ELBO and Adam (Bayes by Backprop + LRT)
    criterion = common.metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss, train_acc, train_kl = common.train_model(net, optimizer, criterion, train_loader,
                                                             num_ens=train_ens, beta_type=beta_type, epoch=epoch,
                                                             num_epochs=epochs)
        valid_loss, valid_acc = common.validate_model(net, criterion, valid_loader, num_ens=valid_ens,
                                                      beta_type=beta_type, epoch=epoch, num_epochs=epochs)
        lr_sched.step(valid_loss)
        print('Epoch:%d, TrainLoss:%.3f, TrainAcc:%.3f, ValLoss:%.3f, ValAcc:%.3f, KL:%.3f' % (
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))
    # Sample k models from the posterior
    nn_dicts = []
    for i in range(K):
        sample_model = net_class(outputs, inputs, layer_type, activation_type).to(device)
        sample_model.sample(net)
        nn_dicts += [sample_model.state_dict()]
    # Save the models
    torch.save(nn_dicts, modelname)
    print("Saved %d models" % K)

# Train dataset
train_dataset = trainset

dirs    = [d for d in os.listdir(HOME_DIR + "mnist_adv_CNN_itcm_%s/" % EPS) if d.startswith("train_images_%s" % COMB)]
dir_ord = sorted([int(re.findall("[0-9]+",d)[0]) for d in dirs])
dirs    = sorted(dirs, key=lambda x: dir_ord.index(int(re.findall("[0-9]+",x)[0])))

transform_back = transforms.Compose([transforms.Resize((28,28))])


images  = []
targets = []
for d in dirs:
    with open(HOME_DIR + "mnist_adv_CNN_itcm_%s/" % EPS + d, 'rb') as handle:
        temp = pickle.load(handle)
        images.append(temp["images"])
        targets.append(temp["labels"])

images  = torch.vstack(images)
images  = transform_back(images)
targets = torch.tensor(sum([t.detach().tolist() for t in targets],[]))
print(images.shape)

# images  = images.permute(0,2,3,1).numpy()
# targets = torch.hstack(targets).numpy()

print("Adversarial Data size: %s images" % images.shape[0])

train_dataset.data    = torch.vstack([train_dataset.data, images])
train_dataset.targets = torch.hstack([train_dataset.targets, targets])

train, val = random_split(train_dataset,[train_dataset.data.shape[0]-10000,10000], generator=torch.Generator().manual_seed(156))

# Train data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val  , batch_size=128, shuffle=True)

train_and_save_models(epochs = 100, K = 100, modelname = HOME_DIR+"/models/lenet-mnist-adv-%s-%s.pt" % (COMB,EPS), force_train=False)
