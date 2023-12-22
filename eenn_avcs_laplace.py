import torch
import numpy as np
import matplotlib.pyplot as plt

from utils_eenn_avcs import *
from utils_notebook import get_logits_targets_image_net
from models.msdnet import MSDNet_exit
from dataloader import get_dataloaders

from laplace import Laplace
# from utils_eenn_avcs import FullLLLaplaceBridge

import random

import pickle

import robustness_metrics as rm



logits, targets, ARGS = get_logits_targets_image_net(model_path='models/image_net', 
                                                     model_name='checkpoint_089.pth.tar',
                                                     root_path="/ssdstore/ImageNet")
probs = torch.softmax(logits, dim=2)
preds = get_preds_per_exit(probs)
acc = get_acc_per_exit(preds, targets)

eces_rm = []
for l in range(L):
    ece = rm.metrics.ExpectedCalibrationError(num_bins=15)
    ece.add_batch(probs[l, :, :].numpy(), label=targets.numpy())
    eces_rm.append(ece.result()['ece'])

print('ECEs: ', eces_rm)
print('Accs: ', acc)