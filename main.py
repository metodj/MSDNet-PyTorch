#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
from functools import partial

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
from utils_poe import schedule_T, get_grad_stats, get_temp_diff_labels, ModifiedSoftmaxCrossEntropyLoss, \
                            CustomBaseCrossEntropyLoss, ModifiedSoftmaxCrossEntropyLossProd
from utils import AverageMeter
from torch.nn.utils import clip_grad_norm_

args = arg_parser.parse_args()

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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import wandb
import numpy as np

torch.manual_seed(args.seed)

os.environ["WANDB_API_KEY"] = "e31842f98007cca7e04fd98359ea9bdadda29073"

def main():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        print(args.save)
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del(model)
        
        
    model = getattr(models, args.arch)(args)


    wandb_kwargs = {
        'project': 'anytime-poe-msdnet',
        'entity': 'metodj',
        'notes': '',
        'mode': 'online',
        'config': vars(args)
    }
    with wandb.init(**wandb_kwargs) as run:
        print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(),
              torch.cuda.get_device_name(0))
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'nr. of trainable params: {params}')
        print(args)

        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

        if args.likelihood == 'softmax':
            if args.loss_type == 'relu':
                criterion = ModifiedSoftmaxCrossEntropyLoss().cuda()
            if args.loss_type == 'relu_prod':
                criterion = ModifiedSoftmaxCrossEntropyLossProd().cuda()
            elif args.loss_type == 'base_a':
                criterion = CustomBaseCrossEntropyLoss().cuda()
            elif args.loss_type == 'standard':
                criterion = nn.CrossEntropyLoss().cuda()
            else:
                raise ValueError()
        elif args.likelihood == 'OVR':
            criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()
        else:
            raise ValueError()

        wandb.watch(model, log='gradients', log_freq=500)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        if args.resume:
            checkpoint = load_checkpoint(args)
            if checkpoint is not None:
                args.start_epoch = checkpoint['epoch'] + 1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

        cudnn.benchmark = True

        train_loader, val_loader, test_loader = get_dataloaders(args)

        if args.evalmode is not None:
            state_dict = torch.load(args.evaluate_from)['state_dict']
            model.load_state_dict(state_dict)

            if args.evalmode == 'anytime':
                # TODO: step in case args.likelihood=='OVR'
                validate(test_loader, model, criterion, args.num_classes, args.likelihood, step=1.)
            else:
                dynamic_evaluate(model, test_loader, val_loader, args)
            return

        scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
                  '\tval_prec1\ttrain_prec5\tval_prec5']

        if args.schedule_T_type == 'sigmoid':
            steps_schedule_T = args.epochs * math.ceil(len(train_loader.dataset) / args.batch_size)
            fun_schedule_T = partial(schedule_T, n_steps=steps_schedule_T, T_start=args.schedule_T_start, T_end=args.schedule_T_end)
        elif args.schedule_T_type == 'constant':
            fun_schedule_T = lambda x: args.schedule_T_start
        else:
            raise ValueError()

        _step = 0
        train_prec1 = None
        for epoch in range(args.start_epoch, args.epochs):

            train_loss, train_prec1, train_prec5, lr, _step, \
                T, train_loss_ind, train_loss_prod, \
                grad_mean, grad_std = train(train_loader, model, criterion, optimizer, epoch,
                                                           args.num_classes, args.likelihood, _step,
                                                           fun_schedule_T, args.alpha, args.ensemble_type, 
                                                           train_prec1, C_mono=args.C_mono, mono_penal=args.mono_penal, 
                                                           stop_grad=args.stop_grad, temp_diff=args.temp_diff, 
                                                           clip_grad=args.clip_grad, loss_type=args.loss_type)
        
            run.log({'train_loss': train_loss.avg})
            run.log({'train_prec1': train_prec1[-1].avg})
            for j in range(args.nBlocks):
                run.log({f'train_prec1_block_{j}': train_prec1[j].avg})
            run.log({'train_loss_ind': train_loss_ind})
            run.log({'train_loss_prod': train_loss_prod})
            run.log({'grad_mean': grad_mean})
            run.log({'grad_std': grad_std})
            run.log({'T': T})
            run.log({'lr': lr})

            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion,
                                                      args.num_classes, args.likelihood, _step, fun_schedule_T, loss_type=args.loss_type)

            run.log({'val_loss': val_loss})
            run.log({'val_prec1': val_prec1})

            scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                          .format(epoch, lr, train_loss.avg, val_loss,
                                  train_prec1[-1].avg, val_prec1, train_prec5, val_prec5))

            is_best = val_prec1 > best_prec1
            if is_best:
                best_prec1 = val_prec1
                best_epoch = epoch
                print('Best var_prec1 {}'.format(best_prec1))

            if (epoch + 1) % 10 == 0:
                model_filename = 'checkpoint_%03d.pth.tar' % epoch
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, args, is_best, model_filename, scores)

        print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

        ### Test the final model

        print('********** Final prediction results **********')
        validate(test_loader, model, criterion, args.num_classes, args.likelihood, _step, fun_schedule_T)

        return

def train(train_loader, model, criterion, optimizer, epoch, num_classes, likelihood, step, step_func=None, 
          alpha=0., ensemble_type="DE", train_prec1=None, C_mono=0., mono_penal=0., stop_grad=False, temp_diff=False, clip_grad=0., loss_type='standard'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # useful for monitoring PoE training
    losses_individual = AverageMeter()
    losses_prod = AverageMeter()

    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    grad_mean, grad_std = None, None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(device=None)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if not isinstance(output, list):
            output = [output]
    
        # noise = 3.
        # output = [x + noise for x in output]

        target_var = get_temp_diff_labels(target_var, output, temp_diff)
        
        loss = 0.0
        L = len(output)
        T = 1.
        for j in range(L):
            if loss_type != 'relu_prod':
                _logits = output[j]
            else:
                # stop_grad for previous logits
                if stop_grad:
                    _logits = torch.stack([output[i].detach() for i in range(j)] + [output[j]], dim=0) 
                else:
                    _logits = torch.stack(output[:j+1], dim=0) 

            loss += criterion(_logits, target_var[j])
            

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if i == 0:
            grad_mean, grad_stat = get_grad_stats(model)

        if clip_grad > 0.:
            clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.4f}\t'
                  'Acc@5 {top5.val:.4f}\t'
                   'T: {T}'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1], T=T))
        step += 1

    return losses, top1, top5[-1].avg, running_lr, step, T, losses_individual.avg, losses_prod.avg, grad_mean, grad_stat

def validate(val_loader, model, criterion, num_classes, likelihood, step, step_func=None, loss_type='standard'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            # TODO: align validation loss with train loss above
            for j in range(len(output)):
                if likelihood == 'softmax':
                    T = 1.
                    if loss_type != 'relu_prod':
                        _logits = output[j]
                    else:
                        # stop_grad for previous logits
                        _logits = torch.stack([output[i].detach() for i in range(j)] + [output[j]], dim=0) 
                    loss += criterion(_logits, target_var)
                elif likelihood == 'OVR':
                    if step_func is not None:
                        T = step_func(step)
                    else:
                        T = 1.
                    loss += criterion(T * output[j], nn.functional.one_hot(target_var, num_classes=num_classes).float())

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}\t'
                      'T: {T}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1], T=T))
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
