import os
from args import arg_parser
import torch


def parse_args():
    args = arg_parser.parse_args(args=[])

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    return args


def schedule_T(step, n_steps, T_start=2., T_end=16.):
    return T_start + (T_end - T_start) * torch.sigmoid(-torch.tensor((-step + n_steps/2) / (n_steps/10)))
