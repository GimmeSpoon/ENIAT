import itertools
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, TypeVar, Union

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.cuda.amp import GradScaler

from ..core import CourseBook, Trainer
from ..utils import StateLogger, advance, bar, end, init_logger, reset
from .grader import TorchGrader
from .learner import TorchLearner
from .predictor import TorchPredictor, distributed, to_tensor

T = TypeVar("T", bound=TorchLearner)
C = TypeVar("C", bound=CourseBook)
L = TypeVar("L", bound=StateLogger)
G = TypeVar("G", bound=TorchGrader)


class TorchTrainer(TorchPredictor, Trainer):
    def __init__(
        self,
        conf: DictConfig = None,
        log=None,
        grader=None,
        course=None,
        learner=None,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
    ) -> None:

        super(TorchTrainer, self).__init__(conf.env)
        self.conf = conf if "trainer" not in conf else conf.trainer
        if log is not None:
            self.log = log
        elif "logger" in conf:
            self.log = init_logger()
        self.grader = grader
        self.course = course
        self.learner = learner
        self.loss = loss_fn
        self.opt = optimizer
        self.sch = scheduler

        warnings.showwarning = lambda *args: self.log.warning(args[0])

    def _save_model(self, timestep: int = None) -> None:
        Path(os.path.join(self.conf.output_dir, "checkpoints")).mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            self.learner.model.state_dict(),
            os.path.join(
                self.conf.output_dir,
                f"checkpoints/model_{timestep}.cpt"
                if timestep
                else "checkpoints/model.cpt",
            ),
        )

    def _save_state(self, timestep: int, filename: str = None) -> None:
        train_state = {}
        train_state["scheduler"] = (
            self.sch.state_dict() if self.sch is not None else None
        )
        train_state["optimizer"] = self.opt.state_dict()
        train_state["unit"] = self.conf.scheme.unit
        train_state["rng_state"] = self.get_rand_state()
        train_state["timestep"] = timestep
        train_state["batch_size"] = self.conf.loader.batch_size
        train_state["total_iters"] = self.conf.scheme.total_iters
        train_state["env"] = self.conf.env
        Path(os.path.join(self.conf.output_dir, "checkpoints")).mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            train_state,
            os.path.join(
                self.conf.output_dir,
                f"checkpoints/state_{timestep}.cpt" if filename is None else filename,
            ),
        )

    def _save_checkpoint(
        self, timestep: int, unit: Literal["epoch", "step"], force: bool = False
    ) -> None:
        if not force and (
            self.conf.scheme.unit != unit or (timestep % self.conf.scheme.save_interval)
            if self.conf.scheme.save_interval
            else True
        ):
            return
        self._save_model(timestep)
        self._save_state(timestep)
        self.log.info(f"Checkpoint at {unit} {timestep} saved.")

    def __chk(self, epoch: int = None, step: int = None):
        if self.conf.scheme.unit == "step" and step is not None:
            if step % self.conf.scheme.save_interval == 0:
                if dist.is_initialized():
                    self.opt.consolidate_state_dict(0)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self._save_checkpoint(step, "step")
        elif self.conf.scheme.unit == "epoch" and epoch is not None:
            if epoch % self.conf.scheme.save_interval == 0:
                if dist.is_initialized():
                    self.opt.consolidate_state_dict(0)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self._save_checkpoint(epoch, "epoch")

    @distributed
    def fit(
        self,
        device: int = 0,
        global_rank: int = 0,
        silent: bool = False,
        final: bool = True,
    ):

        resumed = self.prepare(
            device,
            log=self.log,
        )

        total_step = 0

        loss = None
        s_loss = torch.zeros(2, device=device, requires_grad=False)
        e_loss = torch.zeros(2, device=device, requires_grad=False)

        if resumed:

            if self.conf.scheme.unit == "step":
                total_step = self.conf.scheme.init_iter

            self.log.info(
                f"Training state ({self.conf.scheme.init_iter} \
                          {self.conf.scheme.unit}) resumed."
            )

        with bar(
            "train",
            total_epochs=self.conf.scheme.total_iters
            if self.conf.scheme.unit == "epoch"
            else None,
            total_steps=self.conf.scheme.total_iters
            if self.conf.scheme.unit == "step"
            else len(self.loader),
            start_epoch=self.conf.scheme.init_iter
            if self.conf.scheme.unit == "epoch"
            else None,
            start_step=self.conf.scheme.init_iter
            if self.conf.scheme.unit == "step"
            else None,
        ) if (not dist.is_initialized() or dist.get_rank() == 0) else nullcontext():

            for epoch in (
                range(self.conf.scheme.init_iter + 1, self.conf.scheme.total_iters + 1)
                if self.conf.scheme.unit == "epoch"
                else itertools.count(1)
            ):

                if self.conf.scheme.unit == "epoch":
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        reset("step")

                for current_step, batch in enumerate(self.loader):

                    total_step += 1

                    # Forward
                    batch = to_tensor(
                        batch, dtype=self.conf.scheme.dtype, device=device
                    )
                    self.learner.model.train(True)
                    self.opt.zero_grad()

                    if self.conf.scheme.precision is None:
                        pred = self.learner.fit(batch[0], device, self.log)
                        cur_loss = self.loss(pred, batch[1].to(device))
                    else:
                        with torch.autocast(
                            torch.device(device),
                            dtype=getattr(torch, self.conf.scheme.precision),
                        ):
                            pred = self.learner.fit(batch[0], device, self.log)
                            cur_loss = self.loss(pred, batch[1].to(device))

                    _back = False

                    # Backward
                    if self.conf.scheme.update_interval:
                        if loss is None:
                            loss = cur_loss
                        else:
                            loss = loss + cur_loss

                        if (current_step + 1) % self.conf.scheme.update_interval:
                            _back = True
                    else:
                        loss = cur_loss
                        _back = True

                    if _back or current_step == len(self.loader) - 1:

                        s_loss[1] = batch[0].size(dim=0)
                        s_loss[0] = loss.item() * s_loss[1]

                        e_loss[0] = e_loss[0] + s_loss[0]
                        e_loss[1] = e_loss[1] + s_loss[1]

                        if self.conf.scheme.gradient_scale:
                            scaler = GradScaler(loss)
                            scaler.scale(loss).backward()
                            scaler.step(self.opt)
                            scaler.update()
                        else:
                            scaler = None
                            loss.backward()
                            self.opt.step()

                        loss = None

                    self.learner.model.train(False)

                    if dist.is_initialized():
                        dist.reduce(s_loss, 0, dist.ReduceOp.SUM)

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        self.log.log_state(
                            {
                                "Training Loss": (s_loss[0] / s_loss[1]).item(),
                                "Learning Rate": np.array(self.sch.get_last_lr())
                                if self.sch
                                else 0,
                            },
                            epoch,
                            total_step,
                            "step",
                        )

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        advance("step")

                    # Evaluation (step)
                    if self.grader is not None:
                        _rng_state = self.get_rand_state()
                        if self.grader.eval(
                            device,
                            global_rank,
                            silent,
                            False,
                            learner=self.learner,
                            timestep=total_step,
                            unit="step",
                            env_conf=self.conf.env,
                        ):
                            self.set_rand_state(_rng_state)

                    # Checkpoint (step)
                    self.__chk(step=total_step)

                    if self.conf.scheme.unit == "step":
                        if self.sch:
                            self.sch.step()
                        if self.conf.scheme.total_iters <= total_step:
                            break

                # STEP OVER ========================================

                if (
                    self.conf.scheme.unit == "step"
                    and self.conf.scheme.total_iters <= total_step
                ):
                    break

                # Scheduler
                if self.sch and self.conf.scheme.unit == "epoch":
                    self.sch.step()

                if dist.is_initialized():
                    dist.reduce(e_loss, 0, dist.ReduceOp.SUM)

                if not dist.is_initialized() or dist.get_rank() == 0:
                    self.log.log_state(
                        {
                            "Training Loss": (e_loss[0] / e_loss[1]).item(),
                            "Learning Rate": np.array(self.sch.get_last_lr())
                            if self.sch
                            else 0,
                        },
                        epoch,
                        total_step,
                        "epoch",
                    )

                e_loss[0] = e_loss[1] = 0

                if not dist.is_initialized() or dist.get_rank() == 0:
                    advance("epoch")

                self.__chk(epoch=epoch)

                # Evaluation (epoch)
                if self.grader is not None:
                    _rng_state = self.get_rand_state()
                    if self.grader.eval(
                        device,
                        global_rank,
                        silent,
                        False,
                        learner=self.learner,
                        timestep=epoch,
                        unit="epoch",
                        env_conf=self.conf.env,
                    ):
                        self.set_rand_state(_rng_state)

        # loop over
        if not dist.is_initialized() or dist.get_rank() == 0:
            end("train")

        if dist.is_initialized():
            self.opt.consolidate_state_dict()

        if not dist.is_initialized() or dist.get_rank() == 0:
            self._save_checkpoint(
                self.conf.scheme.total_iters, self.conf.scheme.unit, True
            )

        if dist.is_initialized():
            dist.destroy_process_group()
