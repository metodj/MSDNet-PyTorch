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


# 1) Load model, evaluate on ImageNet
logits, targets, ARGS = get_logits_targets_image_net(model_path='models/image_net/save_models', 
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

# 2) Initialize Laplace
L = 5

models_exit = []
for l in range(L):
    model_l, ARGS = init_model("ImageNet", 'image_net', 'softmax', "089", MSDNet_exit, exit=l+1, cuda=True, ARGS=ARGS)
    models_exit.append(model_l)



# 3) get 2 random disjoint subsets of data of size N each
val_loader = get_image_net_val_loader(ARGS)

N_all = len(val_loader.dataset)
N = 5000

subset1 = random.sample(range(N_all), N)
subset2 = random.sample([i for i in range(N_all) if i not in subset1], N)
subset1 = Subset(val_loader.dataset, subset1)
subset2 = Subset(val_loader.dataset, subset2)

subset1_loader = DataLoader(subset1, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.workers, pin_memory=True)
subset2_loader = DataLoader(subset2, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.workers, pin_memory=True)

print(len(val_loader), len(subset1_loader), len(subset2_loader))


# 4) Optimize prior precision
# TODO

# 5) Fit Laplace
LAs = []
for l in range(L):
    print("Fitting Laplace: ", l)
    la = Laplace(models_exit[l], 'classification', subset_of_weights='last_layer', hessian_structure='full')
    la.fit(subset1_loader)
    # la.prior_precision = LAs_prior_precision[l]
    LAs.append(la)


# 6) Get Laplace predictions
pred_type='glm'
# pred_type='nn'
link_approx='mc'
# link_approx='bridge_norm'
n_samples_num=10

probs_num = []
for l in range(L):
    probs_num_l = []
    targets = []
    with torch.no_grad():
        for i, (x, y) in enumerate(subset2_loader):
            
            y = y.cuda()
            x = x.cuda()

            input_var = torch.autograd.Variable(x)
            target_var = torch.autograd.Variable(y)

            output_num = LAs[l](input_var, n_samples=n_samples_num, pred_type=pred_type, link_approx=link_approx)

            probs_num_l.append(output_num)
            targets.append(target_var)

    probs_num_l = torch.cat(probs_num_l, dim=0).cpu()
    targets = torch.cat(targets).cpu()
    probs_num.append(probs_num_l)

probs_num = torch.stack(probs_num, dim=0)

preds = get_preds_per_exit(probs_num)
acc = get_acc_per_exit(preds, targets)
print(acc)


# 7) Get Credible Sets
# TODO


# store probs_num as pickle
with open('probs_num.pkl', 'wb') as f:
    pickle.dump(probs_num, f)

# store probs as pickle
with open('probs.pkl', 'wb') as f:
    pickle.dump(probs, f)

# store targets as pickle
with open('targets.pkl', 'wb') as f:
    pickle.dump(targets, f)