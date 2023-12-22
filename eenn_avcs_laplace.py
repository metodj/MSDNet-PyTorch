import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from utils_eenn_avcs import *
from utils_notebook import get_logits_targets_image_net, get_image_net_val_loader
from models.msdnet import MSDNet_exit
from dataloader import get_dataloaders

from laplace import Laplace
# from utils_eenn_avcs import FullLLLaplaceBridge

import random

import pickle

# import robustness_metrics as rm

print('Hello World')


# Load model, evaluate on ImageNet
logits, targets, ARGS = get_logits_targets_image_net(model_path='models/image_net', 
                                                     model_name='checkpoint_089.pth.tar',
                                                     root_path="/ssdstore/ImageNet")
probs = torch.softmax(logits, dim=2)
preds = get_preds_per_exit(probs)
acc = get_acc_per_exit(preds, targets)

# eces_rm = []
# for l in range(L):
#     ece = rm.metrics.ExpectedCalibrationError(num_bins=15)
#     ece.add_batch(probs[l, :, :].numpy(), label=targets.numpy())
#     eces_rm.append(ece.result()['ece'])

# print('ECEs: ', eces_rm)
print('Accs: ', acc)

# Initialize Laplace
L = 5

models_exit = []
for l in range(L):
    model_l, ARGS = init_model("ImageNet", 'image_net', 'softmax', 89, MSDNet_exit, exit=l+1, cuda=True)
    models_exit.append(model_l)


# Optimize prior precision
val_loader = get_image_net_val_loader(ARGS)

N_all = len(val_loader.dataset)
N = 5000

# get 2 random disjoint subsets of data of size N each
subset1 = random.sample(range(N_all), N)
subset2 = random.sample([i for i in range(N_all) if i not in subset1], N)
subset1 = Subset(val_loader.dataset, subset1)
subset2 = Subset(val_loader.dataset, subset2)

subset1_loader = DataLoader(subset1, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.workers, pin_memory=True)
subset2_loader = DataLoader(subset2, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.workers, pin_memory=True)

print(len(val_loader), len(subset1_loader), len(subset2_loader))