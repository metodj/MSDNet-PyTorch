import torch
import numpy as np
from models.msdnet import MSDNet, MSDNet_exit
from dataloader import get_dataloaders
from utils import parse_args

from collections import OrderedDict
import random
from typing import List, Union

from laplace.lllaplace import FullLLLaplace


def get_preds_per_exit(probs: torch.Tensor):
    L = probs.shape[0]
    return {i: torch.argmax(probs, dim=2)[i, :] for i in range(L)}


def get_acc_per_exit(
    preds, targets: torch.Tensor
):
    L = len(preds)
    return [(targets == preds[i]).sum() / len(targets) for i in range(L)]


def init_model(dataset, model_folder, likelihood, epoch, model_class, exit=None, cuda=True, ARGS=None):
    assert dataset in ["cifar10", "cifar100", "ImageNet"]
    if ARGS is None:
        assert dataset in ["cifar10", "cifar100"]
        ARGS = parse_args()
        ARGS.data_root = "data"
        ARGS.data = dataset
        if dataset == "cifar10":
            folder_path = 'models_cifar_10'
        else:
            folder_path = 'models'
        ARGS.save = (
            # f"/home/metod/Desktop/PhD/year1/PoE/MSDNet-PyTorch/{folder_path}/{model_folder}"
            f"/home/mona/Desktop/MSDNet-PyTorch/{folder_path}/{model_folder}"
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
    else:
        folder_path = "models"

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



class FullLLLaplaceBridge(FullLLLaplace):
    def __call__(self, x, pred_type='glm', joint=False, link_approx='probit', 
                 n_samples=100, diagonal_output=False, generator=None):
        """Compute the posterior predictive on input data `x`.

        Parameters
        ----------
        x : torch.Tensor
            `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        link_approx : {'mc', 'probit', 'bridge', 'bridge_norm'}
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only 'mc' is possible. 

        joint : bool
            Whether to output a joint predictive distribution in regression with
            `pred_type='glm'`. If set to `True`, the predictive distribution
            has the same form as GP posterior, i.e. N([f(x1), ...,f(xm)], Cov[f(x1), ..., f(xm)]).
            If `False`, then only outputs the marginal predictive distribution. 
            Only available for regression and GLM predictive.

        n_samples : int
            number of samples for `link_approx='mc'`.

        diagonal_output : bool
            whether to use a diagonalized posterior predictive on the outputs.
            Only works for `pred_type='glm'` and `link_approx='mc'`.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used).

        Returns
        -------
        predictive: torch.Tensor or Tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax).
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
            For `likelihood='regression'` and `joint=True`, a tuple of torch.Tensor 
            is returned with the mean and the predictive covariance. 
        """
        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if link_approx not in ['mc', 'probit', 'bridge', 'bridge_norm']:
            raise ValueError(f'Unsupported link approximation {link_approx}.')

        if pred_type == 'nn' and link_approx != 'mc':
            raise ValueError('Only mc link approximation is supported for nn prediction type.')
        
        if generator is not None:
            if not isinstance(generator, torch.Generator) or generator.device != x.device:
                raise ValueError('Invalid random generator (check type and device).')

        if pred_type == 'glm':
            f_mu, f_var = self._glm_predictive_distribution(
                x,
            )
            # regression
            if self.likelihood == 'regression':
                return f_mu, f_var
            # classification
            if link_approx == 'mc':
                return self.predictive_samples(x, pred_type='glm', n_samples=n_samples,).mean(dim=0)
            elif link_approx == 'probit':
                kappa = 1 / torch.sqrt(1. + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
                return torch.softmax(kappa * f_mu, dim=-1)
            elif 'bridge' in link_approx:
                # zero mean correction
                f_mu -= (f_var.sum(-1) * f_mu.sum(-1).reshape(-1, 1) /
                         f_var.sum(dim=(1, 2)).reshape(-1, 1))
                f_var -= (torch.einsum('bi,bj->bij', f_var.sum(-1), f_var.sum(-2)) /
                          f_var.sum(dim=(1, 2)).reshape(-1, 1, 1))
                # Laplace Bridge
                _, K = f_mu.size(0), f_mu.size(-1)
                f_var_diag = torch.diagonal(f_var, dim1=1, dim2=2)
                # optional: variance correction
                if link_approx == 'bridge_norm':
                    f_var_diag_mean = f_var_diag.mean(dim=1)
                    f_var_diag_mean /= torch.as_tensor([K/2], device=self._device).sqrt()
                    f_mu /= f_var_diag_mean.sqrt().unsqueeze(-1)
                    f_var_diag /= f_var_diag_mean.unsqueeze(-1)
                sum_exp = torch.exp(-f_mu).sum(dim=1).unsqueeze(-1)
                alpha = (1 - 2/K + f_mu.exp() / K**2 * sum_exp) / f_var_diag
                # return torch.nan_to_num(alpha / alpha.sum(dim=1).unsqueeze(-1), nan=1.0)
                return alpha
        else:
            samples = self._nn_predictive_samples(x, n_samples)
            if self.likelihood == 'regression':
                return samples.mean(dim=0), samples.var(dim=0)
            return samples.mean(dim=0)
        

def avcs_classification(
    dirichlet_alphas: Union[np.array, torch.Tensor],
    alpha: float = 0.05,
    S: int = 1,
    seed: int = 0,
) -> List[List[int]]:
    np.random.seed(seed)

    L, C = dirichlet_alphas.shape
    alpha_sums = dirichlet_alphas.sum(axis=1)

    # Vectorized sampling
    lik_samples = np.array(
        [np.random.dirichlet(dirichlet_alphas[l]) for l in range(L) for _ in range(S)]
    )
    lik_samples = lik_samples.reshape(L, S, C)

    R_denom = lik_samples.min(axis=1)
    R_arr = (dirichlet_alphas / alpha_sums[:, None]) / R_denom

    R_arr = R_arr.cumprod(axis=0)
    C_arr = [list(np.where(row <= 1 / alpha)[0]) for row in R_arr]

    C_arr = running_intersection_classification(C_arr)
    return C_arr


def targets_dir_alphas(targets, C, eps=0.001, precision=1e2):
    """
    Convert targets to Dirichlet alphas.
    
    Arguments:
    - targets: Tensor of shape (N,) containing class indices.
    - C: Number of classes.
    
    Returns:
    - Tensor of shape (N, C) with one-hot encoding.
    """
    one_hot = torch.zeros(targets.size(0), C, device=targets.device) + eps
    one_hot.scatter_(1, targets.unsqueeze(1), 1 - eps * (C - 1))
    return one_hot * precision


def KL_dirichlet(alpha, beta):
    """
    KL_dirichlet(alpha, beta) calculates the Kullback-Leibler divergence
    between two Dirichlet distributions with parameters alpha and beta
    respectively for each sample in the batch. It is assumed that the parameters are valid, 
    that is alpha[i] > 0 and beta[i] > 0 for all i in range(len(alpha)).
    
    Parameters:
    - alpha (torch.Tensor): parameters of the first Dirichlet distribution for each sample in the batch.
                            Shape: [batch_size, feature_dim]
    - beta (torch.Tensor): parameters of the second Dirichlet distribution for each sample in the batch.
                            Shape: [batch_size, feature_dim]

    Returns:
    - D (torch.Tensor): the KL divergence for each sample in the batch.
                        Shape: [batch_size]
    """
    
    # Summation along the feature dimension
    sum_alpha = torch.sum(alpha, dim=1)
    sum_beta = torch.sum(beta, dim=1)
    
    D = (torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
         - torch.sum(torch.lgamma(alpha), dim=1)
         + torch.sum(torch.lgamma(beta), dim=1)
         + torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha).unsqueeze(1)), dim=1))
    
    return D