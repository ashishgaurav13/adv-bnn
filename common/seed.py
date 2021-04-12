import torch
import numpy as np
import pyro
import warnings 

# Set all seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    warnings.filterwarnings('ignore')
