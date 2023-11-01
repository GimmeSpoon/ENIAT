from typing import TypeVar, Literal
from .grader import TorchGrader
from .predictor import TorchPredictor, to_tensor, distributed
from ..core import Trainer, CourseBook
from .learner import TorchLearner
from ..utils import StateLogger, bar, advance, reset, end
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.live import Live
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import os
from omegaconf import DictConfig
from pathlib import Path
import warnings
from contextlib import nullcontext

T = TypeVar("T", bound=TorchLearner)
C = TypeVar("C", bound=CourseBook)
L = TypeVar("L", bound=StateLogger)
G = TypeVar("G", bound=TorchGrader)


def load_trainer(cfg: DictConfig, log: L, grader: G = None, course=None, learner=None):

    # instantiate trainer
    trainer = TorchTrainer(cfg, log=log, grader=grader, course=course, learner=learner)

    if trainer:
        log.info("Trainer instance created.")

    return trainer


class TorchTrainer(TorchPredictor, Trainer):
    r"""PyTorch compatible trainer class.
    Automatically manage training steps, logging, and saving checkpoints.
    It only occupies one device(GPU)"""

    def __init__(
        self, conf: DictConfig = None, log=None, grader=None, course=None, learner=None
    ) -> None:

        super(TorchTrainer, self).__init__(conf.env)
        self.conf = conf
        self.log = log
        self.grader = grader
        self.course = course
        self.learner = learner

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
            self.learner.sch.state_dict() if self.learner.sch is not None else None
        )
        train_state["optimizer"] = self.learner.opt.state_dict()
        train_state["unit"] = self.conf.scheme.unit
        train_state["rng_state"] = self.get_rand_state()
        train_state["timestep"] = timestep
        train_state["batch_size"] = self.conf.loader.batch_size
        train_state["maxstep"] = self.conf.scheme.max_step
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
                    self.learner.opt.consolidate_state_dict(0)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self._save_checkpoint(step, "step")
        elif self.conf.scheme.unit == "epoch" and epoch is not None:
            if epoch % self.conf.scheme.save_interval == 0:
                if dist.is_initialized():
                    self.learner.opt.consolidate_state_dict(0)
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
        saved = False

        loss = None
        s_loss = torch.zeros(2, device=device, requires_grad=False)
        e_loss = torch.zeros(2, device=device, requires_grad=False)

        if resumed:
            self.set_rand_state(resumed["rng_state"])
            self.conf.scheme.init_step = resumed["timestep"]
            self.conf.scheme.unit = resumed["unit"]

            if self.conf.scheme.unit == "step":
                total_step = self.conf.scheme.init_step

            self.log.info("Training state resumed.")

        with bar(
            "train",
            total_epochs=self.conf.scheme.max_step
            if self.conf.scheme.unit == "epoch"
            else None,
            total_steps=self.conf.scheme.max_step
            if self.conf.scheme.unit == "step"
            else len(self.loader),
            start_epoch=self.conf.scheme.init_step
            if self.conf.scheme.unit == "epoch"
            else None,
            start_step=self.conf.scheme.init_step
            if self.conf.scheme.unit == "step"
            else None,
        ) if (not dist.is_initialized() or dist.get_rank() == 0) else nullcontext():

            for epoch in range(
                self.conf.scheme.init_step + 1,
                self.conf.scheme.max_step + 1
                if self.conf.scheme.unit == "epoch"
                else 2,
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
                    self.learner.opt.zero_grad()

                    if self.conf.scheme.precision is None:
                        cur_loss = self.learner.fit(batch, device, self.log)
                    else:
                        with torch.autocast(
                            torch.device(device),
                            dtype=getattr(torch, self.conf.scheme.precision),
                        ):
                            cur_loss = self.learner.fit(batch, device, self.log)

                    _back = False

                    # Backward
                    if self.conf.scheme.update_interval:
                        if loss is None:
                            loss = cur_loss
                        else:
                            loss += cur_loss

                        if (current_step + 1) % self.conf.scheme.update_interval:
                            _back = True
                    else:
                        loss = cur_loss
                        _back = True

                    if _back:
                        if self.conf.scheme.gradient_scale:
                            scaler = GradScaler(loss)
                            scaler.scale(loss).backward()
                            scaler.step(self.learner.opt)
                            scaler.update()
                        else:
                            scaler = None
                            loss.backward()
                            self.learner.opt.step()

                    self.learner.model.train(False)

                    s_loss[1] = batch[0].size(dim=0)
                    s_loss[0] = loss.item() * s_loss[1]

                    e_loss[0] += s_loss[0]
                    e_loss[1] += s_loss[1]

                    if dist.is_initialized():
                        dist.reduce(s_loss, 0, dist.ReduceOp.SUM)

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        self.log.log_state(
                            {
                                "Training Loss": (s_loss[0] / s_loss[1]).item(),
                                "Learning Rate": self.learner.sch.get_lr()
                                if self.learner.sch
                                else None,
                            },
                            epoch,
                            current_step + 1,
                            "step",
                        )

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        advance("step")

                    # Evaluation (step)
                    if self.grader is not None:
                        _rng_state = self.get_rand_state()
                        self.grader.eval(
                            device,
                            global_rank,
                            silent,
                            False,
                            learner=self.learner,
                            timestep=total_step,
                            unit="step",
                            env_conf=self.conf.env,
                        )
                        self.set_rand_state(_rng_state)

                    # Checkpoint (step)
                    self.__chk(step=total_step)

                # Scheduler
                if self.learner.sch:
                    self.learner.sch.step()

                if dist.is_initialized():
                    dist.reduce(e_loss, 0, dist.ReduceOp.SUM)

                if not dist.is_initialized() or dist.get_rank() == 0:
                    self.log.log_state(
                        {
                            "Training Loss": (e_loss[0] / e_loss[1]).item(),
                            "Learning Rate": self.learner.sch.get_lr()
                            if self.learner.sch
                            else None,
                        },
                        epoch,
                        current_step + 1,
                        "epoch",
                    )

                e_loss[0] = e_loss[1] = 0

                if not dist.is_initialized() or dist.get_rank() == 0:
                    advance("epoch")

                self.__chk(epoch=epoch)

                # Evaluation (epoch)
                if self.grader is not None:
                    _rng_state = self.get_rand_state()
                    self.grader.eval(
                        device,
                        global_rank,
                        silent,
                        False,
                        learner=self.learner,
                        timestep=epoch,
                        unit="epoch",
                        env_conf=self.conf.env,
                    )
                    self.set_rand_state(_rng_state)

        # loop over
        if not dist.is_initialized() or dist.get_rank() == 0:
            end("train")

        if dist.is_initialized():
            self.learner.opt.consolidate_state_dict()

        if not dist.is_initialized() or dist.get_rank() == 0:
            self._save_checkpoint(self.conf.scheme.max_step, self.conf.scheme.unit)

        if dist.is_initialized():
            dist.destroy_process_group()
