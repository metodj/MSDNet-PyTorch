import torch
import numpy as np
from typing import Dict
import scipy
from tqdm import tqdm
from dataloader import get_dataloaders
from models.msdnet import MSDNet
from utils import parse_args
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# TODO: most of the functions below have an ugly implementation with for loops, vectorize them

def modal_probs_decreasing(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int, verbose: bool = False, N: int = 10000) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime
    """
    nr_non_decreasing = {-0.01: 0, -0.05: 0, -0.1: 0, -0.2: 0, -0.5: 0}
    diffs = []
    for i in range(N):
        probs_i = _probs[:, i, _preds[layer - 1][i]].cpu().numpy()
        diffs_i = np.diff(probs_i)
        diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {-1. * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()}
    return nr_decreasing


def modal_probs_average(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int, N: int) -> Dict[float, float]:
    """
    average modal probability in anytime-prediction regime
    """
    preds = []
    for i in range(N):
        preds.append(_probs[:, i, _preds[layer - 1][i]])
    return torch.stack(preds, dim=1).mean(axis=1)


def ovr_likelihood(tensor: torch.Tensor) -> torch.tensor:
    tensor = torch.clone(tensor)
    # TODO: implement without for loops
    assert len(tensor.shape) == 3  # (L, N_test, C)
    for l in range(tensor.shape[0]):
        for n in range(tensor.shape[1]):
            ovr_mask = tensor[l, n, :] > 0
            nr_non_zero = ovr_mask.sum()
            if nr_non_zero == 0:
                nr_non_zero = 1  # to avoid division by 0
            tensor[l, n, :] = ovr_mask.long() / nr_non_zero
    return tensor


def f_probs_ovr_poe(tensor: torch.Tensor) -> torch.tensor:
    tensor = torch.clone(tensor)
    # TODO: implement without for loops
    assert len(tensor.shape) == 3  # (L, N_test, C)
    tensor = tensor.cumprod(dim=0)
    for l in range(tensor.shape[0]):
        for n in range(tensor.shape[1]):
            ovr_mask = tensor[l, n, :] > 0
            if ovr_mask.sum() > 0:
                tensor[l, n, :] = ovr_mask.long() / ovr_mask.sum()
    return tensor


def f_probs_ovr_poe_break_ties(logits, probs_ovr_poe, T=1., softmax=False, sigmoid=False):
    logits, probs_ovr_poe = torch.clone(logits), torch.clone(probs_ovr_poe)
    # TODO: implement without for loops
    assert len(logits.shape) == 3  # (L, N_test, C)
    assert len(probs_ovr_poe.shape) == 3  # (L, N_test, C)
    preds = []
    for l in tqdm(range(logits.shape[0])):
        preds_l = []
        for n in tqdm(range(logits.shape[1])):
            ovr_mask = probs_ovr_poe[l, n, :] > 0
            preds_n = []
            sum_n = 0.
            for c in range(logits.shape[2]):
                if ovr_mask[c]:
                    if sigmoid:
                        sigmoid_prod = torch.sigmoid(T * logits[:l + 1, n, c]).prod()
                    else:
                        sigmoid_prod = (T * logits[:l + 1, n, c]).prod()
                else:
                    sigmoid_prod = 0.
                sum_n += sigmoid_prod
                preds_n.append(sigmoid_prod)
            if sum_n > 0.:
                if softmax:
                    preds_n = list(scipy.special.softmax(preds_n))
                else:
                    preds_n = [x / sum_n for x in preds_n]
            preds_l.append(preds_n)
        preds.append(preds_l)
    return torch.tensor(preds)

# TODO: unit test checking that f_probs_ovr_poe_logits_weighted is the same as f_probs_ovr_poe_break_ties

def f_probs_ovr_poe_logits_weighted(logits, threshold=0.):
    C = logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = (probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2))
    return probs


def f_probs_ovr_break_ties(logits, probs_ovr, T=1.):
    logits, probs_ovr = torch.clone(logits), torch.clone(probs_ovr)
    # TODO: implement without for loops
    assert len(logits.shape) == 3  # (L, N_test, C)
    assert len(probs_ovr.shape) == 3  # (L, N_test, C)
    preds = []
    for l in range(logits.shape[0]):
        preds_l = []
        for n in range(logits.shape[1]):
            ovr_mask = probs_ovr[l, n, :] > 0
            preds_n = []
            sum_n = 0.
            for c in range(logits.shape[2]):
                if ovr_mask[c]:
                    sigmoid_prod = (T * logits[l, n, c])
                else:
                    sigmoid_prod = 0.
                sum_n += sigmoid_prod
                preds_n.append(sigmoid_prod)
            if sum_n > 0.:
                preds_n = [x / sum_n for x in preds_n]
            preds_l.append(preds_n)
        preds.append(preds_l)
    return torch.tensor(preds)


def f_probs_ovr_logits_weighted(logits):
    C = logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < 0] = 0.
    # normalize
    probs = (probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2))
    return probs


def f_preds_ovr_fallback_ood(probs: torch.Tensor, logits: torch.Tensor, prod: bool = False) -> torch.Tensor:
    """
    Predictions for OVR likelihood that fallback to logits when probabilities collapse to 0 (OOD)
    """
    preds = []
    for l in range(probs.shape[0]):
        preds_l = []
        for n in range(probs.shape[1]):
            if probs[l, n, :].sum() > 0.:
                preds_l.append(torch.argmax(probs[l, n, :]))
            else:
                if prod:
                    preds_l_n = torch.argmax(torch.sigmoid(logits[:, n, :]).cumprod(dim=0)[l, :])
                else:
                    preds_l_n = torch.argmax(logits[l, n, :])
                preds_l.append(preds_l_n)
        preds.append(preds_l)
    return torch.tensor(preds)


def f_probs_ovr_fallback_ood(probs: torch.Tensor, logits: torch.Tensor, prod: bool = False) -> torch.Tensor:
    """
    Probabilities for OVR likelihood that fallback to logits when probabilities collapse to 0 (OOD)
    """
    preds = []
    for l in range(probs.shape[0]):
        preds_l = []
        for n in range(probs.shape[1]):
            if probs[l, n, :].sum() > 0.:
                preds_l.append(probs[l, n, :])
            else:
                if prod:
                    preds_l_n = torch.sigmoid(logits[:, n, :]).cumprod(dim=0)[l, :]
                else:
                    preds_l_n = logits[l, n, :]
                preds_l_n = preds_l_n / preds_l_n.sum()  # normalization
                preds_l.append(preds_l_n)
        preds.append(torch.stack(preds_l, dim=0))
    return torch.stack(preds, dim=0)
    

def get_ood_ovr(_probs: torch.Tensor, L: int = 7) -> Dict:
    ood_dict_ids = {l: [] for l in range(L)}
    for n in range(10000):
        for l in range(L):
            nr_non_zero = (_probs[l, n, :] > 0).sum()
            if nr_non_zero == 0:
                ood_dict_ids[l].append(n)
    return ood_dict_ids


def f_probs_ovr_poe_logits_weighted_generalized(logits, threshold=0., weights=None):
    L, C = logits.shape[0], logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = (probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2))
    return probs


def get_logits_targets(dataset, model_folder, likelihood, epoch):
    assert dataset in ['cifar10', 'cifar100']
    ARGS = parse_args()
    ARGS.data_root = 'data'
    ARGS.data = dataset
    ARGS.save= f'/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/models/{model_folder}'
    ARGS.arch = 'msdnet'
    ARGS.batch_size = 64
    ARGS.epochs = 300
    ARGS.nBlocks = 7
    ARGS.stepmode = 'even'
    ARGS.base = 4
    ARGS.nChannels = 16
    ARGS.j = 16
    ARGS.num_classes = 100 if dataset == 'cifar100' else 10
    ARGS.step = 2
    ARGS.use_valid = True
    ARGS.splits = ['train', 'val', 'test']
    ARGS.likelihood = likelihood

    # load pre-trained model
    model = MSDNet(args=ARGS)
    model_path = f'models/{model_folder}/save_models/checkpoint_{epoch}.pth.tar'
    state = torch.load(model_path)
    params = OrderedDict()
    for params_name, params_val in state['state_dict'].items():
        params[params_name.replace('module.', '')] = params_val
        # state['state_dict'][params_name.replace('module.', '')] = state['state_dict'].pop(params_name)
    model.load_state_dict(params)
    model = model.cuda()
    model.eval()

    # data
    _, _, test_loader = get_dataloaders(ARGS)

    logits = []
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

            logits.append(torch.stack(output))
            targets.append(target_var)

    logits = torch.cat(logits, dim=1).cpu()
    targets = torch.cat(targets).cpu()

    return logits, targets, ARGS


def get_logits_targets_image_net(step=4):
    assert step in [4, 7]
    if step == 4:
        ARGS = parse_args()
        ARGS.data_root = 'data'
        ARGS.data = 'ImageNet'
        ARGS.save= f'/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/image_net'
        ARGS.arch = 'msdnet'
        ARGS.batch_size = 64
        ARGS.epochs = 90
        ARGS.nBlocks = 5
        ARGS.stepmode = 'even'
        ARGS.base = 4
        ARGS.nChannels = 32
        ARGS.growthRate = 16
        ARGS.bnFactor = [1, 2, 4, 4]
        ARGS.grFactor = [1, 2, 4, 4]
        ARGS.j = 16
        ARGS.num_classes = 1000
        ARGS.step = 4
        ARGS.use_valid = True
        ARGS.splits = ['train', 'val', 'test']
        ARGS.likelihood = 'softmax'
        ARGS.nScales = len(ARGS.grFactor)
        MODEL_PATH = f'image_net/msdnet-step=4-block=5.pth.tar'
    elif step == 7:
        ARGS = parse_args()
        ARGS.data_root = 'data'
        ARGS.data = 'ImageNet'
        ARGS.save= f'/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/image_net'
        ARGS.arch = 'msdnet'
        ARGS.batch_size = 64
        ARGS.epochs = 90
        ARGS.nBlocks = 5
        ARGS.stepmode = 'even'
        ARGS.base = 7
        ARGS.nChannels = 32
        ARGS.growthRate = 16
        ARGS.bnFactor = [1, 2, 4, 4]
        ARGS.grFactor = [1, 2, 4, 4]
        ARGS.j = 16
        ARGS.num_classes = 1000
        ARGS.step = 7
        ARGS.use_valid = True
        ARGS.splits = ['train', 'val', 'test']
        ARGS.likelihood = 'softmax'
        ARGS.nScales = len(ARGS.grFactor)
        MODEL_PATH = f'image_net/msdnet-step=7-block=5.pth.tar'

    DATA_PATH = 'data/image_net/valid/'

    # load pre-trained model
    model = MSDNet(args=ARGS)
    state = torch.load(MODEL_PATH)
    params = OrderedDict()
    for params_name, params_val in state['state_dict'].items():
        params[params_name.replace('module.', '')] = params_val
        # state['state_dict'][params_name.replace('module.', '')] = state['state_dict'].pop(params_name)
    model.load_state_dict(params)
    model = model.cuda()
    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'nr. of trainable params: {params}')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    val_set = datasets.ImageFolder(DATA_PATH, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]))

    val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=ARGS.batch_size, shuffle=False,
                num_workers=ARGS.workers, pin_memory=True)


    logits = []
    targets = []
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(val_loader)):
            y = y.cuda(device=None)
            x = x.cuda()

            input_var = torch.autograd.Variable(x)
            target_var = torch.autograd.Variable(y)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            logits.append(torch.stack(output))
            targets.append(target_var)

    logits = torch.cat(logits, dim=1).cpu()
    targets = torch.cat(targets).cpu()

    return logits, targets, ARGS


def get_scale_probs(probs_name_arr, probs_arr, T_arr, targets, C, L):

    def scale_probs(_probs, _targets, T):
        probs_scaled = _probs ** T
        probs_scaled = probs_scaled / np.repeat(probs_scaled.sum(axis=2)[:, :, np.newaxis], C, axis=2)
        preds_scaled = {i: torch.argmax(probs_scaled, dim=2)[i, :] for i in range(L)}
        acc_scaled = [(_targets == preds_scaled[i]).sum() / len(_targets) for i in range(L)]
        return probs_scaled, preds_scaled, acc_scaled

    scaled_dict = {x: {} for x in probs_name_arr}
    for probs_name, probs_arr in zip(probs_name_arr, probs_arr):
        for T in T_arr:
            scaled_dict[probs_name][T] = scale_probs(probs_arr, targets, T)
    
    return scaled_dict