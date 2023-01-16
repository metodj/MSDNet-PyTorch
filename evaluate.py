import torch
import numpy as np
from dataloader import get_dataloaders
from models.msdnet import MSDNet
from utils import parse_args
from collections import OrderedDict
import matplotlib.pyplot as plt

args = parse_args()
args.data_root = 'data'
args.data = 'cifar100'
args.save= '/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/models/models'
args.arch = 'msdnet'
args.batch_size = 64
args.epochs = 300
args.nBlocks = 7
args.stepmode = 'even'
args.base = 4
args.nChannels = 16
args.j = 16

print(args)

# load pre-trained model
model = MSDNet(args=args)
MODEL_PATH = 'models/models/save_models/checkpoint_299.pth.tar'
state = torch.load(MODEL_PATH)
params = OrderedDict()
for params_name, params_val in state['state_dict'].items():
    params[params_name.replace('module.', '')] = params_val
    # state['state_dict'][params_name.replace('module.', '')] = state['state_dict'].pop(params_name)
model.load_state_dict(params)
model = model.cuda()
model.eval()

# data
_, _, test_loader = get_dataloaders(args)

preds = []
targets = []
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        y = y.cuda(device=None)
        x = x.cuda()

        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(y)

        output = model(input_var)
        if not isinstance(output, list):
            output = [output]

        preds.append(torch.stack(output))
        targets.append(target_var)

preds = torch.cat(preds, dim=1)
probs = torch.softmax(preds, dim=2)
max_index = torch.argmax(probs, dim=2)[-1, :]

# sanity check that fitted model is used
targets = torch.cat(targets)
print((targets == max_index).sum() / len(targets))


# print(preds.shape)
# print(probs[5, 5, :].sum())
# print(probs[:, 5, 5].sum())
# print(probs[5, :, 5].sum())
# print(max_index.shape)
# print(max_index)

# # PLOT
# for i in range(10):
#     probs_i = probs[:, i, max_index[i]].cpu()
#     print(probs_i)
#     plt.plot(list(range(len(probs_i))), probs_i, label=f'{i}')
# plt.savefig('/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/modal_probs.pdf')
# plt.show()

# nr. of non-decreasing probability vectors in anytime-prediction regime
nr_non_decreasing = 0
diffs = []
for i in range(10000):
    probs_i = probs[:, i, max_index[i]].cpu().numpy()
    diffs_i = np.diff(probs_i)
    diffs.append(diffs_i.min())
    if np.all(diffs_i >= -0.1):
        nr_non_decreasing += 1
print(nr_non_decreasing)
print(np.mean(diffs))

# # deep ensembles
# assert torch.all(probs[0, 0, :] == probs.cumsum(dim=0)[0, 0, :])
# de_probs = probs.cumsum(dim=0).cpu() / torch.tensor([1., 2., 3., 4., 5., 6., 7.])[:, None, None]
# de_max_index = torch.argmax(de_probs, dim=2)[-1, :]
# # print(de_probs.shape)
# # print(de_max_index.shape)
# # print(de_probs[0, 0, :].sum())
# print((targets.cpu() == de_max_index.cpu()).sum() / len(targets))
# nr_non_decreasing = 0
# diffs = []
# for i in range(10000):
#     probs_i = de_probs[:, i, de_max_index[i]].cpu().numpy()
#     diffs_i = np.diff(probs_i)
#     diffs.append(diffs_i.min())
#     if np.all(diffs_i >= -0.05):
#         nr_non_decreasing += 1
# print(nr_non_decreasing)
# print(np.mean(diffs))

