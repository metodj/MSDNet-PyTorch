import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


def get_cascade_dynamic_weights(train_prec: Union[List[AverageMeter], None], L: int, weight_type: str = 'depth'):
    if train_prec is None:
        return [1. for _ in range(L)]
    assert len(train_prec) == L
    weights = [1 / train_prec[l].avg for l in range(L)]
    if weight_type == 'softmax':
        return np.exp(weights) / sum(np.exp(weights))
    elif weight_type == 'sum':
        w_sum = sum(weights)
        return [w / w_sum for w in weights]
    elif weight_type == 'depth':
        L_sum = L * (L + 1) / 2
        return [l / L_sum for l in range(1, L + 1)]
    elif weight_type == 'uniform':
        return [1. for _ in range(L)]
    else:
        raise ValueError()
    

def get_mono_weights(output: List[torch.Tensor], targets: torch.Tensor, C_base: float = 1., C_mono: float = 1.) -> torch.Tensor:
    """
    stop gradient is implemented to avoid backpropagating through the weights

    stop gradient is implemented via .detach : 
        https://stackoverflow.com/questions/51529974/tensorflow-stop-gradient-equivalent-in-pytorch
    """
    _output = [o.detach() for o in output]  # stop gradient
    L = len(_output)
    b = _output[0].shape[0]
    w =  b * torch.ones(L, device=_output[0].device)

    w[0] *= C_base
    for l in range(1, L):
        _preds_prev, _preds = torch.argmax(_output[l - 1], dim=1), torch.argmax(_output[l], dim=1)
        mask = (_preds_prev == targets) & (_preds != targets)
        w[l] += mask.sum() * C_mono

    w = w / b
    return w


def cross_entropy_loss_manual(logits: torch.Tensor, targets: torch.Tensor, loss_type: str = 'one_minus', reduction: str = 'mean', stop_grad: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes cross entropy loss with reversed probabilities
    """
    if stop_grad:
        logits = logits.detach()
    if loss_type == 'original':
        probs = torch.nn.functional.softmax(logits, dim=1)
    elif loss_type == 'one_minus':
        probs = 1. - torch.nn.functional.softmax(logits, dim=1)
    else:
        raise ValueError()
    probs = torch.gather(probs, 1, targets.unsqueeze(1))
    loss = -torch.log(probs + eps)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError()
    

def get_temp_diff_labels(target_var: torch.Tensor, output: List[torch.Tensor], temp_diff: bool) -> List[torch.Tensor]:
    if temp_diff:
        return [out.argmax(dim=1).detach() for out in output[1:]] + [target_var]
    else:
        return [target_var for _ in range(len(output))]
    

class ModifiedSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-2):
        super(ModifiedSoftmaxCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, target):
        # Apply the modified softmax based on ReLU
        modified_softmax = torch.relu(logits) / (torch.sum(torch.relu(logits), dim=1, keepdim=True) + self.eps)

        target_one_hot = torch.zeros_like(modified_softmax).scatter_(1, target.unsqueeze(1), 1)

        # Calculate the negative log-likelihood loss
        loss = -torch.sum(target_one_hot * torch.log(modified_softmax + self.eps), dim=1)  # Add epsilon for numerical stability
        # loss = -torch.sum(target_one_hot * modified_softmax + eps, dim=1)  # Add epsilon for numerical stability

        # Calculate the average loss across the batch
        return torch.mean(loss)


class ModifiedSoftmaxCrossEntropyLossProd(nn.Module):
    def __init__(self, eps=1e-4, eps_log=1e-20, act_func='relu'):
        super(ModifiedSoftmaxCrossEntropyLossProd, self).__init__()
        self.eps = eps
        self.eps_log = eps_log
        assert act_func in ['relu', 'softplus']
        if act_func == 'relu':
            self.act_func = torch.relu
        elif act_func == 'softplus':
            self.act_func = F.softplus
        

    # TODO: explore further the effect of eps
    def forward(self, logits, target):

        # Apply activation function to logits
        logits = self.act_func(logits)

        # Take product along axis=0
        prod_logits = torch.prod(logits, dim=0)

        # Normalize prod_logits to get valid probabilities
        modified_softmax = prod_logits / (torch.sum(prod_logits, dim=1, keepdim=True) + self.eps)

        # Create one-hot representation of target
        target_one_hot = torch.zeros_like(modified_softmax).scatter_(1, target.unsqueeze(1), 1)

        # Calculate the negative log-likelihood loss
        loss = -torch.sum(target_one_hot * torch.log(modified_softmax + self.eps_log), dim=1)  # Add epsilon for numerical stability
        # loss = -torch.sum(target_one_hot * modified_softmax, dim=1)  # Add epsilon for numerical stability

        # Calculate the average loss across the batch
        return torch.mean(loss)
    

class CustomBaseCrossEntropyLoss(nn.Module):
    def __init__(self, a=1.2):
        super(CustomBaseCrossEntropyLoss, self).__init__()
        assert a > 0, "The base 'a' must be greater than 0"
        self.a = a

    def forward(self, logits, target, eps=1e-9):
        # Subtract the maximum logit for numerical stability (log-sum-exp trick)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]

        # Calculate the custom base exponential of the logits
        custom_exp = torch.pow(self.a, logits)

        # Normalize custom_exp to create a probability distribution
        custom_softmax = custom_exp / torch.sum(custom_exp, dim=1, keepdim=True)

        # Calculate the negative log-likelihood loss
        target_one_hot = torch.zeros_like(custom_softmax).scatter_(1, target.unsqueeze(1), 1)
        loss = -torch.sum(target_one_hot * torch.log(custom_softmax + eps), dim=1)

        # Calculate the average loss across the batch
        return torch.mean(loss)
    

# # Define the custom loss function
# criterion = ModifiedSoftmaxCrossEntropyLoss()
# _criterion = nn.CrossEntropyLoss()
# a_base_criterion = CustomBaseCrossEntropyLoss(a=1.1)

# # set torch random seed
# torch.manual_seed(0)

# # Assume logits and target are tensors representing the predicted logits and true labels
# logits = torch.randn(4, 10)  # A batch of 4 samples with 10 classes each
# target = torch.randint(0, 10, (4,))  # A batch of 4 ground-truth class labels

# # Calculate the loss
# loss = criterion(logits, target)
# _loss = _criterion(logits, target)
# a_base_loss = a_base_criterion(logits, target)

# print(loss)
# print(_loss)
# print(a_base_loss)


    


