import torch
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader
from typing import Literal
from math import sin, cos, pi
import warnings

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
        self.base_lr = base_lr
        if self.warmup_type == "linear":
            super().__init__(
                optimizer,
                [(lambda x: x * group['initial_lr'] / self.warmup_steps) for group in optimizer.param_groups]
                if base_lr is None else lambda x: x * base_lr / self.warmup_steps,
                last_epoch,
                verbose)
        elif self.warmup_type == "cosine":
            super().__init__(
                optimizer,
                [(lambda x: group['initial_lr'] * (sin(pi * x / self.warmup_steps) + 1) / 2) for group in optimizer.param_groups],
                last_epoch,
                verbose)
        self.base_lr = base_lr

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
        self.base_lr = base_lr
        super().__init__(
            optimizer,
            [(lambda x: (1 - x /self.decay_steps) * group['initial_lr']) for group in optimizer.param_groups],
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
        self.base_lr = base_lr
        super().__init__(
            optimizer,
            [(lambda x: group['initial_lr'] * (cos(pi * x / self.warmup_steps) + 1) / 2) for group in optimizer.param_groups],
            last_epoch,
            verbose)

class LinearDecayWithWarmupScheduler(SequentialLR):
    def __init__(
        self,
        optimizer,
        total_iters:int,
        warmup_steps:int,
        warmup_type:Literal["linear", "cosine"]="linear",
        warmup_unit:Literal["step", "epoch"]="step",
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.warmup_sch = WarmupScheduler(optimizer, base_lr, warmup_steps, warmup_type, last_epoch, verbose)
        self.base_sch = LinearDecayScheduler(optimizer, base_lr, total_iters - warmup_steps, last_epoch, verbose)
        super().__init__(optimizer, [self.warmup_sch, self.base_sch], [warmup_steps], last_epoch, verbose)

class CosineDecayWithWarmupScheduler(SequentialLR):
    def __init__(
        self,
        optimizer,
        total_iters:int,
        warmup_steps:int,
        warmup_type:Literal["linear", "cosine"]="linear",
        warmup_unit:Literal["step", "epoch"]="step",
        base_lr:float = None,
        last_epoch:int = -1,
        verbose:bool = False,
    ) -> None:
        self.warmup_sch = WarmupScheduler(optimizer, base_lr, warmup_steps, warmup_type, last_epoch, verbose)
        self.base_sch = CosineDecayScheduler(optimizer, base_lr, total_iters - warmup_steps, last_epoch, verbose)
        super().__init__(optimizer, [self.warmup_sch, self.base_sch], [warmup_steps], last_epoch, verbose)