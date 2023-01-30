import torch
from typing import List, Union
from utils import AverageMeter


def schedule_T(step, n_steps, T_start=2., T_end=16.):
    return T_start + (T_end - T_start) * torch.sigmoid(-torch.tensor((-step + n_steps/2) / (n_steps/10)))


def get_prod_loss(output: List[torch.Tensor], criterion: torch.nn.modules.loss, num_classes: int, T: float) -> torch.Tensor:
    """
    Computes logZ part of PoE loss
    """
    b = output[0].shape[0]
    criterion.reduction = 'none'

    prod_loss = []
    for c in range(num_classes):
        y = torch.nn.functional.one_hot((torch.ones(b) * c).long(), num_classes=num_classes).float().cuda()
        c_loss = torch.zeros(b, requires_grad=True).cuda()
        for j in range(len(output)):
            c_loss += -criterion(T * output[j], y).mean(axis=1)
        prod_loss.append(c_loss.clone())

    prod_loss = torch.stack(prod_loss, dim=1)
    criterion.reduction = 'mean'
    return torch.sum(torch.logsumexp(prod_loss, dim=1))


def get_grad_stats(model):
    grad = torch.cat([torch.flatten(v.grad.abs()) for _, v in model.named_parameters()])
    return torch.mean(grad), torch.std(grad)

def get_depth_weighted_logits(logits: List[torch.Tensor], depth: int) -> List[torch.Tensor]:
    w = torch.tensor([depth - i for i in range(depth)])
    w = w / w.sum()
    return [w[j] * logits[j] for j in range(depth)]


def get_cascade_dynamic_weights(train_prec: Union[List[AverageMeter], None], L: int):
    if train_prec is None:
        return [1. for _ in range(L)]
    assert len(train_prec) == L
    weights = [1 / train_prec[l].avg for l in range(L)]
    w_sum = sum(weights)
    return [w / w_sum for w in weights]