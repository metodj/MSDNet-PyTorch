import torch
from typing import List


def schedule_T(step, n_steps, T_start=2., T_end=16.):
    return T_start + (T_end - T_start) * torch.sigmoid(-torch.tensor((-step + n_steps/2) / (n_steps/10)))


def get_prod_loss(output: List[torch.Tensor], criterion: torch.nn.modules.loss, num_classes: int, T: float) -> torch.Tensor:
    """
    Computes logZ part of PoE loss
    """
    prod_loss = torch.tensor(0., requires_grad=True).cuda()
    for c in range(num_classes):
        y = torch.nn.functional.one_hot((torch.ones(output[0].shape[0]) * c).long(), num_classes=num_classes).float().cuda()
        c_loss = torch.tensor(0., requires_grad=True).cuda()
        for j in range(len(output)):
            c_loss += -criterion(T * output[j], y)
        prod_loss += torch.exp(c_loss.clone())
    return prod_loss
