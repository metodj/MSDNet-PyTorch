import torch
import numpy as np
from typing import Dict
import scipy
from tqdm import tqdm


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


def modal_probs_average(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int) -> Dict[float, float]:
    """
    average modal probability in anytime-prediction regime
    """
    preds = []
    for i in range(10000):
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