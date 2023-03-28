import torch
import numpy as np
from typing import Dict, Optional, List
import scipy
from tqdm import tqdm
from dataloader import get_dataloaders
from models.msdnet import MSDNet
from utils import parse_args
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict, Counter
from sklearn.linear_model import LogisticRegression


# TODO: most of the functions below have an ugly implementation with for loops, vectorize them


def modal_probs_decreasing(
    _preds: Dict[int, torch.Tensor],
    _probs: torch.Tensor,
    layer: Optional[int] = None,
    verbose: bool = False,
    N: int = 10000,
    diffs_type: str = "consecutive",
    thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.5],
    return_ids: bool = False
) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime

    function can also be used for ground truth probabilities, set layer=None
    """
    nr_non_decreasing = {threshold: 0 for threshold in thresholds}
    diffs = {threshold: [] for threshold in thresholds}
    for i in range(N):
        if layer is None:
            c = _preds[i]
        else:
            c = _preds[layer - 1][i]
        probs_i = _probs[:, i, c].cpu().numpy()
        if diffs_type == "consecutive":
            diffs_i = np.diff(probs_i)
        elif diffs_type == "all":
            diffs_i = probs_decrease(probs_i)
        else:
            raise ValueError()
        # diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                diffs[threshold].append(i)
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {
        -1.0 * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()
    }
    if return_ids:
        return nr_decreasing, diffs
    else:
        return nr_decreasing


def modal_probs_average(
    _preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int, N: int
) -> Dict[float, float]:
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


def f_probs_ovr_poe_break_ties(
    logits, probs_ovr_poe, T=1.0, softmax=False, sigmoid=False
):
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
            sum_n = 0.0
            for c in range(logits.shape[2]):
                if ovr_mask[c]:
                    if sigmoid:
                        sigmoid_prod = torch.sigmoid(T * logits[: l + 1, n, c]).prod()
                    else:
                        sigmoid_prod = (T * logits[: l + 1, n, c]).prod()
                else:
                    sigmoid_prod = 0.0
                sum_n += sigmoid_prod
                preds_n.append(sigmoid_prod)
            if sum_n > 0.0:
                if softmax:
                    preds_n = list(scipy.special.softmax(preds_n))
                else:
                    preds_n = [x / sum_n for x in preds_n]
            preds_l.append(preds_n)
        preds.append(preds_l)
    return torch.tensor(preds)


# TODO: unit test checking that f_probs_ovr_poe_logits_weighted is the same as f_probs_ovr_poe_break_ties


def f_probs_ovr_poe_logits_weighted(logits, threshold=0.0):
    C = logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs

def f_probs_ovr_poe_logits_softmax(logits, L, threshold=0.0):
    L, N, _ = logits.shape
    _logits = logits.numpy().copy()
    _logits[_logits < threshold] = 0.0
    mask = np.cumprod(_logits, axis=0)  > 0.

    # _logits = np.cumsum(_logits, axis=0)
    _logits = np.cumsum(_logits, axis=0) / np.array([float(i) for i in range(1, L + 1)])[:, None, None]
    # _logits = np.cumprod(_logits, axis=0) 

    probs = np.zeros(mask.shape)
    for l in range(L):
        for n in range(N):
            probs[l, n, mask[l, n, :]] = scipy.special.softmax(_logits[l, n, mask[l, n, :]])

    return probs


def f_probs_ovr_break_ties(logits, probs_ovr, T=1.0):
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
            sum_n = 0.0
            for c in range(logits.shape[2]):
                if ovr_mask[c]:
                    sigmoid_prod = T * logits[l, n, c]
                else:
                    sigmoid_prod = 0.0
                sum_n += sigmoid_prod
                preds_n.append(sigmoid_prod)
            if sum_n > 0.0:
                preds_n = [x / sum_n for x in preds_n]
            preds_l.append(preds_n)
        preds.append(preds_l)
    return torch.tensor(preds)


def f_probs_ovr_logits_weighted(logits):
    C = logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < 0] = 0.0
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs


def f_preds_ovr_fallback_ood(
    probs: torch.Tensor, logits: torch.Tensor, prod: bool = False
) -> torch.Tensor:
    """
    Predictions for OVR likelihood that fallback to logits when probabilities collapse to 0 (OOD)
    """
    preds = []
    for l in range(probs.shape[0]):
        preds_l = []
        for n in range(probs.shape[1]):
            if probs[l, n, :].sum() > 0.0:
                preds_l.append(torch.argmax(probs[l, n, :]))
            else:
                if prod:
                    preds_l_n = torch.argmax(
                        torch.sigmoid(logits[:, n, :]).cumprod(dim=0)[l, :]
                    )
                else:
                    preds_l_n = torch.argmax(logits[l, n, :])
                preds_l.append(preds_l_n)
        preds.append(preds_l)
    return torch.tensor(preds)


def f_probs_ovr_fallback_ood(
    probs: torch.Tensor, logits: torch.Tensor, prod: bool = False
) -> torch.Tensor:
    """
    Probabilities for OVR likelihood that fallback to logits when probabilities collapse to 0 (OOD)
    """
    preds = []
    for l in range(probs.shape[0]):
        preds_l = []
        for n in range(probs.shape[1]):
            if probs[l, n, :].sum() > 0.0:
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


def f_probs_ovr_poe_logits_weighted_generalized(logits, threshold=0.0, weights=None):
    L, C = logits.shape[0], logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs


def get_logits_targets(dataset, model_folder, likelihood, epoch, cuda=True, logits_type: str = 'test'):
    assert dataset in ["cifar10", "cifar100"]
    assert logits_type in ['train', 'test', 'val']
    ARGS = parse_args()
    ARGS.data_root = "data"
    ARGS.data = dataset
    ARGS.save = (
        f"/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/models/{model_folder}"
    )
    ARGS.arch = "msdnet"
    ARGS.batch_size = 64
    ARGS.epochs = 300
    ARGS.nBlocks = 7
    ARGS.stepmode = "even"
    ARGS.base = 4
    ARGS.nChannels = 16
    ARGS.j = 16
    ARGS.num_classes = 100 if dataset == "cifar100" else 10
    ARGS.step = 2
    ARGS.use_valid = True
    ARGS.splits = ["train", "val", "test"]
    ARGS.likelihood = likelihood

    # load pre-trained model
    model = MSDNet(args=ARGS)
    model_path = f"models/{model_folder}/save_models/checkpoint_{epoch}.pth.tar"
    if cuda:
        state = torch.load(model_path)
    else: 
        state = torch.load(model_path, map_location=torch.device('cpu'))
    params = OrderedDict()
    for params_name, params_val in state["state_dict"].items():
        params[params_name.replace("module.", "")] = params_val
        # state['state_dict'][params_name.replace('module.', '')] = state['state_dict'].pop(params_name)
    model.load_state_dict(params)
    if cuda:
        model = model.cuda()
    model.eval()

    # data
    if logits_type == 'test':
        _, _, _loader = get_dataloaders(ARGS)
    elif logits_type == 'val':
        _, _loader, _ = get_dataloaders(ARGS)
    elif logits_type == 'train':
        _loader, _, _ = get_dataloaders(ARGS)
    else:
        raise ValueError(f'logits_type={logits_type} not supported')

    logits = []
    targets = []
    with torch.no_grad():
        for i, (x, y) in enumerate(_loader):
            if cuda:
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
        ARGS.data_root = "data"
        ARGS.data = "ImageNet"
        ARGS.save = f"/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/image_net"
        ARGS.arch = "msdnet"
        ARGS.batch_size = 64
        ARGS.epochs = 90
        ARGS.nBlocks = 5
        ARGS.stepmode = "even"
        ARGS.base = 4
        ARGS.nChannels = 32
        ARGS.growthRate = 16
        ARGS.bnFactor = [1, 2, 4, 4]
        ARGS.grFactor = [1, 2, 4, 4]
        ARGS.j = 16
        ARGS.num_classes = 1000
        ARGS.step = 4
        ARGS.use_valid = True
        ARGS.splits = ["train", "val", "test"]
        ARGS.likelihood = "softmax"
        ARGS.nScales = len(ARGS.grFactor)
        MODEL_PATH = f"image_net/msdnet-step=4-block=5.pth.tar"
    elif step == 7:
        ARGS = parse_args()
        ARGS.data_root = "data"
        ARGS.data = "ImageNet"
        ARGS.save = f"/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/image_net"
        ARGS.arch = "msdnet"
        ARGS.batch_size = 64
        ARGS.epochs = 90
        ARGS.nBlocks = 5
        ARGS.stepmode = "even"
        ARGS.base = 7
        ARGS.nChannels = 32
        ARGS.growthRate = 16
        ARGS.bnFactor = [1, 2, 4, 4]
        ARGS.grFactor = [1, 2, 4, 4]
        ARGS.j = 16
        ARGS.num_classes = 1000
        ARGS.step = 7
        ARGS.use_valid = True
        ARGS.splits = ["train", "val", "test"]
        ARGS.likelihood = "softmax"
        ARGS.nScales = len(ARGS.grFactor)
        MODEL_PATH = f"image_net/msdnet-step=7-block=5.pth.tar"

    # load pre-trained model
    model = MSDNet(args=ARGS)
    state = torch.load(MODEL_PATH)
    params = OrderedDict()
    for params_name, params_val in state["state_dict"].items():
        params[params_name.replace("module.", "")] = params_val
        # state['state_dict'][params_name.replace('module.', '')] = state['state_dict'].pop(params_name)
    model.load_state_dict(params)
    model = model.cuda()
    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"nr. of trainable params: {params}")

    val_loader =get_image_net_val_loader(ARGS)

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


def get_image_net_val_loader(ARGS):

    DATA_PATH = "data/image_net/valid/"

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    val_set = datasets.ImageFolder(
        DATA_PATH,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=ARGS.batch_size,
        shuffle=False,
        num_workers=ARGS.workers,
        pin_memory=True,
    )

    return val_loader


def get_scale_probs(probs_name_arr, probs_arr, T_arr, targets, C, L):
    def scale_probs(_probs, _targets, T):
        probs_scaled = _probs**T
        probs_scaled = probs_scaled / np.repeat(
            probs_scaled.sum(axis=2)[:, :, np.newaxis], C, axis=2
        )
        preds_scaled = {i: torch.argmax(probs_scaled, dim=2)[i, :] for i in range(L)}
        acc_scaled = [
            (_targets == preds_scaled[i]).sum() / len(_targets) for i in range(L)
        ]
        return probs_scaled, preds_scaled, acc_scaled

    scaled_dict = {x: {} for x in probs_name_arr}
    for probs_name, probs_arr in zip(probs_name_arr, probs_arr):
        for T in T_arr:
            scaled_dict[probs_name][T] = scale_probs(probs_arr, targets, T)

    return scaled_dict


def probs_decrease(probs: np.array) -> np.array:
    L = len(probs)
    diffs = []
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(probs[j] - probs[i])
    return np.array(diffs)


def get_image_net_val_loader(ARGS, normalize=True):
    DATA_PATH = "data/image_net/valid/"

    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    val_set = datasets.ImageFolder(
        DATA_PATH,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=ARGS.batch_size,
        shuffle=False,
        num_workers=ARGS.workers,
        pin_memory=True,
    )

    return val_loader


def change_your_mind_analysis(names, preds_change_arr, preds_arr, targets, N):
    res_dict = {x: {} for x in names}
    for _type, _preds_change, _preds in zip(names, preds_change_arr, preds_arr):
        count_dict = {'no_change (corr_pred)': 0, 'no_change (incorr_pred)': 0, 'corr_to_incorr': 0, 'incorr_to_corr': 0, 'both': 0}
        for n in range(N):
            _preds_change_n = _preds_change[:, n]
            if np.all(_preds_change_n == 0):
                if _preds[n] == targets[n]:
                    count_dict['no_change (corr_pred)'] += 1
                else:
                    count_dict['no_change (incorr_pred)'] += 1
            elif -1 in _preds_change_n and 1 in _preds_change_n:
                count_dict['both'] += 1
            elif -1 in _preds_change_n and 1 not in _preds_change_n:
                count_dict['incorr_to_corr'] += 1
            elif -1 not in _preds_change_n and 1 in _preds_change_n:
                count_dict['corr_to_incorr'] += 1
            else:
                raise ValueError('Should not happen')

        assert sum(list(count_dict.values())) == N
        res_dict[_type] = count_dict
    return res_dict


def probs_decrease_relative(probs: np.array) -> np.array:
    L = len(probs)
    diffs = []
    for i in range(L):
        for j in range(i + 1, L):
            if probs[i] != 0:
                diffs.append(probs[j]/ probs[i])
            else:
                diffs.append(1.)
    return np.array(diffs) - 1.


def modal_probs_decreasing_relative(
    _preds: Dict[int, torch.Tensor],
    _probs: torch.Tensor,
    layer: Optional[int] = None,
    verbose: bool = False,
    N: int = 10000,
    diffs_type: str = "consecutive",
    thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.5],
) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime

    function can also be used for grount truth probabilities, set layer=None
    """
    nr_non_decreasing = {threshold: 0 for threshold in thresholds}
    # diffs = []
    for i in range(N):
        if layer is None:
            c = _preds[i]
        else:
            c = _preds[layer - 1][i]
        probs_i = _probs[:, i, c].cpu().numpy()
        if diffs_type == "consecutive":
            raise NotImplementedError
        elif diffs_type == "all":
            diffs_i = probs_decrease_relative(probs_i)
        else:
            raise ValueError()
        # diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {
        -1.0 * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()
    }
    return nr_decreasing


def f_probs_ovr_poe_logits_sigmoid(logits, threshold=0.0, min_max_norm=True):
    C = logits.shape[-1]
    _logits = logits.numpy().copy()
    _logits[_logits < threshold] = 0.0
    mask = np.cumprod(_logits, axis=0)  > 0.

    probs = np.zeros(mask.shape)
    for l in range(L):
        for n in range(N):
            _mask_l_n = mask[l, n, :]
            _logits_l_n = _logits[l, n, _mask_l_n]
            if min_max_norm:
                _logits_l_n = (_logits_l_n - _logits_l_n.min()) / (_logits_l_n.max() - _logits_l_n.min())  # for numerical stability
            probs[l, n, _mask_l_n] = scipy.special.expit(_logits_l_n)  # sigmoid
            
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs


def f_probs_ovr_poe_logits_sigmoid_log_probs(logits, threshold=0.0):
    _logits = logits.numpy().copy()  # (L, N, C)
    L, N, _ =  _logits.shape
    _logits[_logits < threshold] = 0.0
    mask = np.cumprod(_logits, axis=0)  > 0.

    probs = np.zeros(mask.shape)
    for l in range(L):
        for n in range(N):
            _mask_l_n = mask[l, n, :]
            _logits_l_n = _logits[:l + 1, n, _mask_l_n]
            _logits_l_n = -np.log(1. + np.exp(-_logits_l_n)).sum(axis=0)  # sum over L
            c_arr = []
            for c in range(len(_logits_l_n)):
                _logits_l_n_c = scipy.special.logsumexp(_logits_l_n - _logits_l_n[c])  # sum over C
                c_arr.append(np.exp(-_logits_l_n_c))
            probs[l, n, _mask_l_n] = c_arr
            
    return probs


def measure_forgetting(preds, targets, L, N):
    forget_ids = []
    for n in range(N):
        for l in range(1, L):

            if preds[l][n] != targets[n] and preds[l - 1][n] == targets[n]:
                forget_ids.append((n, l))
                break

    return [(x / N) * 100 for x in list(Counter([x[1] + 1 for x in forget_ids]).values())]


def f_probs_ovr_poe_logits_weighted_generalized_break_ties(logits, threshold=0.0, weights=None):
    L, N, C = logits.shape[0], logits.shape[1], logits.shape[2]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    for l in range(L):
        for n in range(N):
            sum_l_n = probs[l, n, :].sum()
            if sum_l_n > 0.:
                probs[l, n, :] = probs[l, n, :] / sum_l_n
            else:
                # probs[l, n, :] = (1 / C) * torch.ones(C)
                # probs[l, n, :] = torch.zeros(C)
                probs[l, n, :] = torch.softmax(logits[l, n, :], dim=0)
                # probs[l, n, :] = (logits[:l + 1, n, :] > 0).sum(axis=0) / (logits[:l + 1, n, :] > 0).sum()
    return probs

def get_probs_ovr_poe_w_adaptive_threshold(logits: torch.Tensor, weights: torch.Tensor, metric: str = 'top_2_logits', thres_metric: float = 4., 
                                          thres_hard: float = 0., thres_easy: float = 10., break_ties: bool = True) -> torch.Tensor:
    """
    TODO: Add fallback to logits in case of a collapse to zero distribution
    """
    if metric == 'top_2_logits':
        top_logits = torch.topk(logits, 2, dim=1).values
        l = 0  # we look at the first early exit
        measure = top_logits[l, 0] - top_logits[l, 1] 
    else:
        raise NotImplementedError()

    if  measure > thres_metric:
        _thres = thres_easy
    else:
        _thres = thres_hard
    
    if break_ties:
        return f_probs_ovr_poe_logits_weighted_generalized_break_ties(logits[:, None, :], threshold=_thres, weights=weights)
    else:
        return f_probs_ovr_poe_logits_weighted_generalized(logits[:, None, :], threshold=_thres, weights=weights)


def get_probs_ovr_poe_w_adapt_thres_log_reg(logits: torch.Tensor, weights: torch.Tensor, 
                                            clf: LogisticRegression, C: float, break_ties: bool = True) -> torch.Tensor:
    """
    TODO: Add fallback to logits in case of a collapse to zero distribution

    here we assume logits have shape (L, C), i.e. we have only one example
    """
    l = 0  # we look at the first early exit
    top_logits = torch.topk(logits[l], 2).values.numpy()
    
    thres = clf.predict_proba(top_logits.reshape(1, -1))[:, 1] * C
    thres = thres[0]  # flatten
    
    if break_ties:
        return f_probs_ovr_poe_logits_weighted_generalized_break_ties(logits[:, None, :], threshold=thres, weights=weights)
    else:
        return f_probs_ovr_poe_logits_weighted_generalized(logits[:, None, :], threshold=thres, weights=weights)


def grid_search_adapt_thres(_logits, _targets, clf, c_arr=[0., 5., 10., 15., 20.]):

    L, N = len(_logits), len(_logits[0])

    C_dict = {}
    for c in c_arr:
        probs_poe_ovr_adapt_thres = np.concatenate([np.nan_to_num(get_probs_ovr_poe_w_adapt_thres_log_reg(
                                                                                                        logits=_logits[:, n, :], 
                                                                                                        weights=(np.arange(1, L + 1, 1, dtype=float) / L), 
                                                                                                        clf=clf, 
                                                                                                        C=c)
                                                                ) for n in range(N)], axis=1)
        probs_poe_ovr_adapt_thres = torch.tensor(probs_poe_ovr_adapt_thres)
        preds_poe_ovr_adapt_thres = {i: torch.argmax(probs_poe_ovr_adapt_thres, dim=2)[i, :] for i in range(L)}
        acc_poe_ovr_adapt_thres = [(_targets == preds_poe_ovr_adapt_thres[i]).sum() / len(_targets) for i in range(L)]
        C_dict[c] = (probs_poe_ovr_adapt_thres, preds_poe_ovr_adapt_thres, acc_poe_ovr_adapt_thres)

    probs = torch.softmax(_logits, dim=2)
    preds = {i: torch.argmax(probs, dim=2)[i, :] for i in range(L)}
    acc = [(_targets == preds[i]).sum() / len(_targets) for i in range(L)]

    C_dict['base'] = (probs, preds, acc)

    return C_dict
