# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import random
from collections import defaultdict, deque
import warnings

import numpy as np
import torch
from torch import nn
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t, dist.ReduceOp.SUM)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]


class MetricLogger(object):
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_dict(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

def set_seed(seed):
    seed = seed + get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def load_first_task_model(model_without_ddp, loss_scaler, task_id, args):
    strict = False

    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    elif os.path.isdir(args.resume):
        path = os.path.join(args.resume, f"checkpoint_{task_id}.pth")
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')

    model_ckpt = checkpoint['model']

    if not strict:
        for i in range(1, 6):
            k = f"head.fcs.{i}.weight"
            if k in model_ckpt: del model_ckpt[k]
            k = f"head.fcs.{i}.bias"
            if k in model_ckpt: del model_ckpt[k]
    model_without_ddp.load_state_dict(model_ckpt, strict=strict)
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #args.start_epoch = checkpoint['epoch'] + 1
        #if args.model_ema:
        #    utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        if 'scaler' in checkpoint:
            try:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            except:
                warnings.warn("Could not reload loss scaler, probably because of amp/noamp mismatch")

def change_pos_embed_size(pos_embed, new_size=32, patch_size=16, old_size=224):
    nb_patches = (new_size // patch_size) ** 2
    new_pos_embed = torch.randn(1, nb_patches + 1, pos_embed.shape[2])
    new_pos_embed[0, 0] = pos_embed[0, 0]

    lo_idx = 1
    for i in range(nb_patches):
        hi_idx = lo_idx + old_size // nb_patches
        new_pos_embed[0, i] = pos_embed[0, lo_idx:hi_idx].mean(dim=0)
        lo_idx = hi_idx

    return torch.nn.Parameter(new_pos_embed)

def set_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad
