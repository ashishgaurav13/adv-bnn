import torch, torchvision
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pyro
import tqdm
import os, pickle,sys
import common


# Settings
train = False

# Reproducibility
common.set_seed(156)


class NN(torch.nn.Module):
    def __init__(self, ni, nh, no):
        super(NN, self).__init__()
        self.A = torch.nn.Linear(ni, nh)
        self.relu = torch.nn.ReLU()
        self.B = torch.nn.Linear(nh, no)
    def forward(self, x):
        # Two layer neural network
        x = self.B(self.relu(self.A(x)))
        return x


# Point estimate NN
net = NN(28*28, 1024, 10)



def model(x, y):
    # Put priors on weights and biases
    priors = {
        "A.weight": pyro.distributions.Normal(
            loc=torch.zeros_like(net.A.weight),
            scale=torch.ones_like(net.A.weight),
        ).independent(2),
        "A.bias": pyro.distributions.Normal(
            loc=torch.zeros_like(net.A.bias),
            scale=torch.ones_like(net.A.bias),
        ).independent(1),
        "B.weight": pyro.distributions.Normal(
            loc=torch.zeros_like(net.B.weight),
            scale=torch.ones_like(net.B.weight),
        ).independent(2),
        "B.bias": pyro.distributions.Normal(
            loc=torch.zeros_like(net.B.bias),
            scale=torch.ones_like(net.B.bias),
        ).independent(1),
    }
    # Create a NN module using the priors
    lmodule = pyro.random_module("module", net, priors)
    regressor = lmodule()
    # Do a forward pass on the NN module, i.e. yhat=f(x) and condition on yhat=y
    lhat = torch.nn.LogSoftmax(dim=1)(regressor(x))
    pyro.sample("obs", pyro.distributions.Categorical(logits=lhat).independent(1), obs=y)



softplus = torch.nn.Softplus()
def guide(x, y):
    # Create parameters for variational distribution priors
    Aw_mu = pyro.param("Aw_mu", torch.randn_like(net.A.weight))
    Aw_sigma = softplus(pyro.param("Aw_sigma", torch.randn_like(net.A.weight)))
    Ab_mu = pyro.param("Ab_mu", torch.randn_like(net.A.bias))
    Ab_sigma = softplus(pyro.param("Ab_sigma", torch.randn_like(net.A.bias)))
    Bw_mu = pyro.param("Bw_mu", torch.randn_like(net.B.weight))
    Bw_sigma = softplus(pyro.param("Bw_sigma", torch.randn_like(net.B.weight)))
    Bb_mu = pyro.param("Bb_mu", torch.randn_like(net.B.bias))
    Bb_sigma = softplus(pyro.param("Bb_sigma", torch.randn_like(net.B.bias)))
    # Create random variables similarly to model
    priors = {
        "A.weight": pyro.distributions.Normal(loc=Aw_mu, scale=Aw_sigma).independent(2),
        "A.bias"  : pyro.distributions.Normal(loc=Ab_mu, scale=Ab_sigma).independent(1),
        "B.weight": pyro.distributions.Normal(loc=Bw_mu, scale=Bw_sigma).independent(2),
        "B.bias"  : pyro.distributions.Normal(loc=Bb_mu, scale=Bb_sigma).independent(1),
    }
    # Return NN module from these random variables
    lmodule = pyro.random_module("module", net, priors)
    return lmodule()


# Do stochastic variational inference to find q(w) closest to p(w|D)
svi = pyro.infer.SVI(
    model, guide, pyro.optim.Adam({'lr': 0.01}), pyro.infer.Trace_ELBO(),)


def train_and_save_models(epochs = 10, K = 100, modelname = "model.pt"):
    if os.path.exists(modelname):
        print("File exists")
        return
    # Train with SVI
    for epoch in range(epochs):
        loss = 0.
        for data in train_loader:
            images, labels = data
            images = images.view(-1, 28*28)
            loss += svi.step(images, labels)
        loss /= len(train_loader.dataset)
        print("Epoch %g: Loss = %g" % (epoch, loss))
    # Sample k models from the posterior
    sampled_models = [guide(None, None) for i in range(K)]
    # Save the models
    nn_dicts = []
    for i in range(len(sampled_models)):
        nn_dicts += [sampled_models[i].state_dict()]
    torch.save(nn_dicts, modelname)
    print("Saved %d models" % K)


def load_models(K = 100):
    # Load the models
    sampled_models = [NN(28*28, 1024, 10) for i in range(K)]
    for net, state_dict in zip(sampled_models, torch.load("/home/fcbeylun/adv-bnn/models/model.pt")):
        net.load_state_dict(state_dict)
    print("Loaded %d sample models" % K)
    return sampled_models


sampled_models = load_models(K = 25)


## Generate Adversarial Examples


# Train dataset
train_dataset = torchvision.datasets.MNIST('/home/fcbeylun/adv-bnn/', train=True, download=False,
                       transform=torchvision.transforms.ToTensor())

# Train data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(156))

# Test dataset
test_dataset = torchvision.datasets.MNIST('/home/fcbeylun/adv-bnn/', train=False, download=False,
                       transform=torchvision.transforms.ToTensor())

# Test data loader with batch_size 1
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(156))



def forward_pass(model, images, loss_target = None):
    output = model(images)
    output = torch.nn.LogSoftmax(dim=-1)(output)
    which_class = torch.argmax(output).item()
    if loss_target:
        loss, target = loss_target
        loss(output, target).backward()
    return which_class


def otcm(images, eps, saliency):
    return torch.clamp(images.clone()-eps*saliency, 0, 1)

# How many models can an adversarial example fool?
def how_many_can_it_fool(sampled_models, eps, saliency,images):
    fool = 0
    for k in range(len(sampled_models)):
        # Forward pass on sampled model k
        old_class = forward_pass(sampled_models[k], images)
        # One step Target Class Method (OTCM); saliency is noise
        new_images = otcm(images, eps, saliency)
        # Forward pass again on adv. example
        new_class = forward_pass(sampled_models[k], new_images)
        # If we change the class, we fool the model
        fool += int(old_class != new_class)
    return fool/len(sampled_models)


def generate_saliency(EPS,target,images):
    # Collect noises (saliencies)
    # EPS = 0.18
    saliencies = []
    how_many_fooled = []
    torch.set_printoptions(sci_mode=False)
    # target = torch.tensor([1])
    target = torch.tensor([target])
    for k in range(len(sampled_models)):
        # Forward pass
        # Compute loss w.r.t. an incorrect class
        # Note that we just have to ensure this class is different from targets
        # print("\r Processing " + str(k+1) + "/%s" % len(sampled_models), end="")
        images.grad = None
        images.requires_grad = True
        old_class = forward_pass(sampled_models[k], images, [torch.nn.NLLLoss(), target])
        # Compute adversarial example
        new_images = otcm(images, EPS, images.grad.sign())
        # Forward pass on adv. example
        new_class = forward_pass(sampled_models[k], new_images)
        if old_class != new_class:
            # How many models can this adv. example fool?
            how_many_fooled += [how_many_can_it_fool(sampled_models, EPS, images.grad.sign(), images)]
            saliencies += [images.grad.sign().view(28, 28)]
    # print("\nFinished")
    return saliencies, how_many_fooled


def combine_saliencies(saliencies,success):
    # distributional saliency map
    saliencies = torch.stack(saliencies)
    # print(saliencies.shape)
    combined_med  = torch.zeros(28, 28)
    combined_mean = torch.zeros(28, 28)
    for i in range(28):
        for j in range(28):
            # choose median perturbation
            combined_med[i, j] = np.percentile(saliencies[:, i, j].numpy(), 50)
            combined_mean[i, j] = saliencies[:, i, j].mean().item()
    combined_med  = combined_med.flatten()
    combined_mean = combined_mean.flatten()
    champ         = saliencies[success.index(max(success))].flatten()
    return combined_med, combined_mean, champ



# Train dataset
train_dataset = torchvision.datasets.MNIST('/home/fcbeylun/adv-bnn/', train=True, download=False,
                       transform=torchvision.transforms.ToTensor())
# Train data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False,
                                          generator=torch.Generator().manual_seed(156))

# Train dataset
test_dataset = torchvision.datasets.MNIST('/home/fcbeylun/adv-bnn/', train=False, download=False,
                       transform=torchvision.transforms.ToTensor())
# Train data loader
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False,
                                          generator=torch.Generator().manual_seed(156))


if train:
    ## Generate
    EPS = 0.18
    SAVE_DIR = "/home/fcbeylun/adv-bnn/mnist_adv/"
    # for i in range(len(train_dataset.targets)):
    target_len = len(train_dataset.classes)
    targets    = set(range(10))
    counter    = 1
    skip       = int(sys.argv[1])*60
    stop       = (int(sys.argv[1])+1)*60
    successes  = []
    for data in train_loader:
        if counter < skip:
            counter +=1
            continue
        images_med   = []
        images_mean  = []
        images_champ = []
        tru_labels   = []
        images, labels = data
        images = images.view(-1, 28*28)
        print("\r Batch %s" % counter, end="")
        for i in range(images.shape[0]): #
            # the real target
            target_org = labels[i].item()
            # the target that wanted to be resulted in
            target     = int(np.random.choice(list(targets - set([target_org])),size=1))
            image      = images[i:i+1,:]
            # generating saliency maps using each sampled network
            temp_sals, success = generate_saliency(EPS,target,image)
            successes.append(success)
            # combining maps into three types
            combined_med, combined_mean, champ = combine_saliencies(temp_sals,success)
            # creating image
            images_med.append(otcm(image, EPS, combined_med))
            images_mean.append(otcm(image, EPS, combined_mean))
            images_champ.append(otcm(image, EPS, champ))
            tru_labels.append(target_org)
        tru_labels   = torch.tensor(tru_labels)

        images_med   = (torch.vstack(images_med).reshape(-1,28, 28)*255).type(torch.uint8).detach()
        images_mean  = (torch.vstack(images_mean).reshape(-1,28, 28)*255).type(torch.uint8).detach()
        images_champ = (torch.vstack(images_champ).reshape(-1,28, 28)*255).type(torch.uint8).detach()
        images_med   = {'images': images_med,  'labels': tru_labels}
        images_mean  = {'images': images_mean, 'labels': tru_labels}
        images_champ = {'images': images_champ,'labels': tru_labels}


        with open(SAVE_DIR + 'train_images_med_%s.pickle'   % counter, 'wb') as handle:
            pickle.dump(images_med, handle, protocol  = pickle.HIGHEST_PROTOCOL)
        with open(SAVE_DIR + 'train_images_mean_%s.pickle'  % counter, 'wb') as handle:
            pickle.dump(images_mean, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open(SAVE_DIR + 'train_images_champ_%s.pickle' % counter, 'wb') as handle:
            pickle.dump(images_champ, handle, protocol= pickle.HIGHEST_PROTOCOL)
        counter += 1
        if counter > stop:
            print("braking")
            break
else:
    ## Generate
    EPS = 0.18
    SAVE_DIR = "/home/fcbeylun/adv-bnn/mnist_adv/"
    # for i in range(len(train_dataset.targets)):
    target_len = len(test_dataset.classes)
    targets    = set(range(10))
    counter    = 1
    skip       = int(sys.argv[1])*60
    stop       = (int(sys.argv[1])+1)*60
    successes  = []
    for data in test_loader:
        if counter < skip:
            counter +=1
            continue
        images_med   = []
        images_mean  = []
        images_champ = []
        tru_labels   = []
        images, labels = data
        images = images.view(-1, 28*28)
        print("\r Batch %s" % counter, end="")
        for i in range(images.shape[0]): #
            # the real target
            target_org = labels[i].item()
            # the target that wanted to be resulted in
            target     = int(np.random.choice(list(targets - set([target_org])),size=1))
            image      = images[i:i+1,:]
            # generating saliency maps using each sampled network
            temp_sals, success = generate_saliency(EPS,target,image)
            successes.append(success)
            # combining maps into three types
            combined_med, combined_mean, champ = combine_saliencies(temp_sals,success)
            # creating image
            images_med.append(otcm(image, EPS, combined_med))
            images_mean.append(otcm(image, EPS, combined_mean))
            images_champ.append(otcm(image, EPS, champ))
            tru_labels.append(target_org)
        tru_labels   = torch.tensor(tru_labels)

        images_med   = (torch.vstack(images_med).reshape(-1,28, 28)*255).type(torch.uint8).detach()
        images_mean  = (torch.vstack(images_mean).reshape(-1,28, 28)*255).type(torch.uint8).detach()
        images_champ = (torch.vstack(images_champ).reshape(-1,28, 28)*255).type(torch.uint8).detach()
        images_med   = {'images': images_med,  'labels': tru_labels}
        images_mean  = {'images': images_mean, 'labels': tru_labels}
        images_champ = {'images': images_champ,'labels': tru_labels}


        with open(SAVE_DIR + 'test_images_med_%s.pickle'   % counter, 'wb') as handle:
            pickle.dump(images_med, handle, protocol  = pickle.HIGHEST_PROTOCOL)
        with open(SAVE_DIR + 'test_images_mean_%s.pickle'  % counter, 'wb') as handle:
            pickle.dump(images_mean, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open(SAVE_DIR + 'test_images_champ_%s.pickle' % counter, 'wb') as handle:
            pickle.dump(images_champ, handle, protocol= pickle.HIGHEST_PROTOCOL)
        counter += 1
        if counter > stop:
            print("braking")
            break
