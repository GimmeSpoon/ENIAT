import torch
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, List, Literal, Union
from math import sin, cos, pi
import warnings
import functools

def linear_warmup_func(timestep:int, initial_lr:float, total_steps:int):
    return (timestep + 1) * initial_lr / total_steps

def linear_decay_func(timestep:int, initial_lr:float, total_steps:int):
    return (total_steps - timestep) * initial_lr / total_steps

def cosine_warmup_func(timestep:int, initial_lr:float, total_steps:int):
    return (sin(timestep / total_steps * pi - pi / 2) + 1) / 2

def cosine_decay_func(timestep:int, initial_lr:float, total_steps:int):
    return (cos(timestep / total_steps * pi) + 1) / 2

class WarmupScheduler(LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps:int,
        warmup_type:Literal["linear", "cosine"]="linear",
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.warmup_type = warmup_type
        self.warmup_steps = warmup_steps
        if self.warmup_type == "linear":
            super().__init__(
                optimizer,
                [ functools.partial(linear_warmup_func, initial_lr=group["initial_lr"], total_steps=warmup_steps) for group in optimizer.param_groups] \
                if base_lr is None else functools.partial(linear_warmup_func, initial_lr=base_lr, total_steps=warmup_steps),
                last_epoch,
                verbose)
        elif self.warmup_type == "cosine":
            super().__init__(
                optimizer,
                [functools.partial(cosine_warmup_func, initial_lr=group["initial_lr"], total_steps=warmup_steps) for group in optimizer.param_groups] \
                if base_lr is None else functools.partial(cosine_warmup_func, initial_lr=base_lr, total_steps=warmup_steps),
                last_epoch,
                verbose)

class LinearDecayScheduler(LambdaLR):
    def __init__(
        self,
        optimizer,
        decay_steps:int,
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.decay_steps = decay_steps
        super().__init__(
            optimizer,
            [functools.partial(linear_decay_func, initial_lr=group["initial_lr"], total_steps=decay_steps) for group in optimizer.param_groups] \
            if base_lr is None else functools.partial(linear_decay_func, initial_lr=base_lr, total_steps=decay_steps),
            last_epoch,
            verbose)

class CosineDecayScheduler(LambdaLR):
    def __init__(
        self,
        optimizer,
        decay_steps:int,
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.decay_steps = decay_steps
        super().__init__(
            optimizer,
            [functools.partial(cosine_decay_func, initial_lr=group["initial_lr"], total_steps=decay_steps) for group in optimizer.param_groups] \
            if base_lr is None else functools.partial(cosine_decay_func, initial_lr=base_lr, total_steps=decay_steps),
            last_epoch,
            verbose)

class LinearDecayWithWarmupScheduler(SequentialLR):
    def __init__(
        self,
        optimizer,
        total_iters:int,
        warmup_steps:int,
        warmup_type:Literal["linear", "cosine"]="linear",
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.warmup_sch = WarmupScheduler(optimizer, warmup_steps, warmup_type, base_lr, last_epoch, verbose)
        self.decay_sch = LinearDecayScheduler(optimizer, total_iters - warmup_steps, base_lr, last_epoch, verbose)
        super().__init__(optimizer, [self.warmup_sch, self.decay_sch], [warmup_steps], last_epoch, verbose)

class CosineDecayWithWarmupScheduler(SequentialLR):
    def __init__(
        self,
        optimizer,
        total_iters:int,
        warmup_steps:int,
        warmup_type:Literal["linear", "cosine"]="linear",
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.warmup_sch = WarmupScheduler(optimizer, warmup_steps, warmup_type, base_lr, last_epoch, verbose)
        self.decay_sch = CosineDecayScheduler(optimizer, total_iters - warmup_steps, base_lr,  last_epoch, verbose)
        super().__init__(optimizer, [self.warmup_sch, self.decay_sch], [warmup_steps], last_epoch, verbose)