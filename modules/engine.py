# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import json
import os
import math
import argparse
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy

from . import utils
import logger as lgr

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, criterion,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, task_id: int, loss_scaler, args,
                    mixup_fn: Optional[Mixup] = None, 
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None):
    model.train()
    # logging start
    lgr.update_state(num_batches=len(data_loader))
    lgr.add_state(total_epochs=1)
    lgr.train_epoch_start()
    metric_logger = utils.MetricLogger()
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1))
    # logging end

    for batch_index, (samples, targets, tasks) in enumerate(data_loader):
        # logging start
        lgr.update_state(batch_num=batch_index)
        lgr.add_state(total_batches=1)
        if batch_index == 0:
            lgr.train_first_batch_start(samples=samples)
        lgr.train_batch_start()
        # logging end

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tasks = tasks.to(device, non_blocking=True)
        optimizer.zero_grad()

        lam = None
        token_tasks = None
        if mixup_fn is not None:
            # lgr.lgrs.loguru.debug(f'before mixup targets: {targets.shape} {targets}')
            samples, targets, lam, token_tasks = mixup_fn(samples, targets, tasks)
            if token_tasks is not None:
                token_tasks = token_tasks.flatten(-2)

            # lgr.lgrs.loguru.debug(f'after mixup targets: {targets.shape} {targets}')
            

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            outputs = model(samples.contiguous())
            loss_dict = model_without_ddp.train_forward(outputs, samples, targets, tasks, task_id, teacher_model, criterion, args, lam=lam, token_tasks=token_tasks)

        loss = sumup_loss(loss_dict)
        # loss = sum(loss_dict.values())
        loss = check_loss(loss)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, model_without_ddp, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # logging start
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], **loss_dict)

        lgr.train_batch_end_with_interval(lr=optimizer.param_groups[0]["lr"], **loss_dict)
        # logging end

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch()
    
    # logging start
    metric_logger.synchronize_between_processes() # average across the entire epoch
    lgr.train_epoch_end(metric_logger=metric_logger)
    # logging end

def sumup_loss(d):
    loss = 0.
    for k, v in d.items():
        if not math.isfinite(v.item()):
            lgr.lgrs.loguru.warning(f'loss item "{k}" is "{v.item()}", ignoring it.')
        else:
            loss += v
    return loss

def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))
    return loss


@torch.no_grad()
def evaluate(data_loader, model, task_id, device):
    # switch to evaluation mode
    model.eval()
    # logging start
    lgr.update_state(num_batches=len(data_loader))
    lgr.eval_epoch_start()
    # logging end

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger()


    for batch_index, (images, target, task_ids) in enumerate(data_loader):
        # logging start
        lgr.update_state(batch_num=batch_index)
        lgr.eval_batch_start(samples=images)
        # logging end

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            logits = output['logits']
            loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, min(5, logits.shape[1])))

        # logging start
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        for i in range(task_id+1):
            mask = task_ids == i
            num_samples = torch.count_nonzero(mask)
            if num_samples == 0:
                continue
            
            task_acc1 = accuracy(logits[mask], target[mask], topk=(1,))[0]
            metric_logger.meters[f'task{i}_acc1'].update(task_acc1.item(), n=num_samples)

        lgr.eval_batch_end_with_interval(loss=loss.item(), acc1=acc1.item(), acc5=acc5.item())
        # logging end

    # logging start
    metric_logger.synchronize_between_processes() # average across the entire epoch

    lgr.eval_epoch_end(metric_logger=metric_logger)


def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets
