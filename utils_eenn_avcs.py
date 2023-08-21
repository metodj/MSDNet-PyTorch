import torch
import numpy as np
from models.msdnet import MSDNet, MSDNet_exit
from dataloader import get_dataloaders
from utils import parse_args

from collections import OrderedDict
import random


def init_model(dataset, model_folder, likelihood, epoch, model_class, exit=None, cuda=True):
    assert dataset in ["cifar10", "cifar100"]
    ARGS = parse_args()
    ARGS.data_root = "data"
    ARGS.data = dataset
    if dataset == "cifar10":
        folder_path = 'models_cifar_10'
    else:
        folder_path = 'models'
    ARGS.save = (
        f"/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/{folder_path}/{model_folder}"
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
    if model_class == MSDNet_exit:
        assert exit
        model = model_class(args=ARGS, exit=exit)
    else:
        model = model_class(args=ARGS)
    model_path = f"{folder_path}/{model_folder}/save_models/checkpoint_{epoch}.pth.tar"
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
    return model, ARGS


def get_logits_targets_exit(model, logits_type, args, cuda=True):
    if logits_type == 'test':
        _, _, _loader = get_dataloaders(args)
    elif logits_type == 'val':
        _, _loader, _ = get_dataloaders(args)
    elif logits_type == 'train':
        _loader, _, _ = get_dataloaders(args)
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

            logits.append(output)
            targets.append(target_var)

    logits = torch.cat(logits, dim=0).cpu()
    targets = torch.cat(targets).cpu()

    return logits, targets


def running_intersection_classification(sets):
    sets_intersect = []
    curr_intersect = set(sets[0])
    for sublist in sets:
        curr_intersect &= set(sublist)
        sets_intersect.append(list(curr_intersect))
    return sets_intersect


def raps_eenn(
    probs: np.ndarray,
    targets: np.ndarray,
    calib_size: float = 0.2,
    alpha: float = 0.05,
    lam_reg: float = 0.01,
    k_reg: float = 5,
    disallow_zero_sets: bool = False,
    rand: bool = True,
    seed: int = 0,
):
    """
    Code adapted from:
        https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    """
    L, N, C = probs.shape

    random.seed(seed)
    calib_ids = random.sample(range(N), int(calib_size * N))
    valid_ids = list(set(range(N)) - set(calib_ids))

    reg_vec = np.array(
        k_reg
        * [
            0,
        ]
        + (C - k_reg)
        * [
            lam_reg,
        ]
    )[None, :]

    sizes, coverages, sets, labels = [], [], [], []
    for exit in range(L):
        cal_smx = probs[exit, calib_ids, :]
        cal_labels = targets[calib_ids].cpu().numpy()
        n = len(cal_labels)

        val_smx = probs[exit, valid_ids, :]
        valid_labels = targets[valid_ids].cpu().numpy()
        n_valid = len(valid_labels)

        # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
        cal_pi = cal_smx.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == cal_labels[:, None])[1]
        cal_scores = (
            cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]
            - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
        )
        # Get the score quantile
        qhat = np.quantile(
            cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
        )
        # Deploy
        n_val = val_smx.shape[0]
        val_pi = val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
        val_srt_reg = val_srt + reg_vec
        indicators = (
            (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val, 1) * val_srt_reg)
            <= qhat
            if rand
            else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        )
        if disallow_zero_sets:
            indicators[:, 0] = True
        conformal_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)

        sizes.append(conformal_sets.sum(axis=1).mean())
        coverages.append(
            conformal_sets[np.arange(n_valid), valid_labels].sum() / n_valid
        )
        sets.append(conformal_sets)
        labels.append(valid_labels)

    return sizes, coverages, sets, labels


def consistency_classifciation(sets) -> float:
    L = len(sets)
    sets_intersect = running_intersection_classification(sets)
    cons_arr = []
    for l in range(L):
        size_intersect = len(sets_intersect[l])
        size = len(sets[l])
        if size_intersect == 0 and size > 0:
            cons_arr.append(0.0)
        elif size_intersect == 0 and size == 0:
            cons_arr.append(1.0)
        elif size_intersect > 0 and size == 0:
            cons_arr.append(1.0)
        else:
            cons_arr.append(size_intersect / size)
    return cons_arr