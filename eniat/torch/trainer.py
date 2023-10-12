from typing import TypeVar, Literal, Callable
from .grader import TorchGrader
from .predictor import TorchPredictor, to_tensor, distributed
from ..core import Trainer, Warning, Course, CourseBook
from .learner import TorchLearner
from ..utils.statelogger import StateLogger
from rich.progress import (Progress, TextColumn, BarColumn,
                           TaskProgressColumn, TimeElapsedColumn,
                           TimeRemainingColumn, MofNCompleteColumn,
                           SpinnerColumn)
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
import os
import random
from omegaconf import DictConfig
from pathlib import Path
import warnings

T = TypeVar('T', bound=TorchLearner)
C = TypeVar('C', bound=CourseBook)
L = TypeVar('L', bound=StateLogger)
G = TypeVar('G', bound=TorchGrader)

def load_trainer(cfg:DictConfig, log:L, grader:G=None, course=None, learner=None):

    # instantiate trainer
    trainer = TorchTrainer(cfg, log=log, grader=grader, course=course, learner=learner)

    if trainer:
        log.info("Trainer instance created.")
    
    return trainer

class TorchTrainer(TorchPredictor, Trainer):
    r"""PyTorch compatible trainer class.
    Automatically manage training steps, logging, and saving checkpoints.
    It only occupies one device(GPU)"""
    def __init__(self, conf:DictConfig=None, log=None, grader=None, course=None, learner=None) -> None:

        super(TorchTrainer, self).__init__(conf.env)
        self.conf = conf
        self.log = log
        self.grader = grader
        self.course = course
        self.learner = learner

        warnings.showwarning = lambda *args: self.log.warning(args[0])

    def _save_model(self, timestep:int=None) -> None:
        Path(os.path.join(self.conf.output_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(self.learner.model.state_dict(), os.path.join( self.conf.output_dir , f'checkpoints/model_{timestep}.cpt' if timestep else 'checkpoints/model.cpt'))

    def _save_state(self, timestep:int, filename:str=None) -> None:
        train_state = {}
        train_state['optimizer'] = self.learner.opt.state_dict()
        train_state['unit'] = self.conf.scheme.unit
        train_state['rng_state'] = self.get_rand_state()
        train_state['timestep'] = timestep
        train_state['batch_size'] = self.conf.loader.batch_size
        train_state['maxstep'] = self.conf.scheme.max_step
        train_state['env'] = self.conf.env
        Path(os.path.join(self.conf.output_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(train_state, os.path.join(self.conf.output_dir, f'checkpoints/state_{timestep}.cpt' if filename is None else filename))

    def _save_checkpoint(self, timestep:int, unit:Literal['epoch', 'step'], force:bool=False) -> None:
        if not force and (self.conf.scheme.unit != unit or (timestep % self.conf.scheme.save_interval) if self.conf.scheme.save_interval else True):
            return
        self._save_model(timestep)
        self._save_state(timestep)
        self.log.info(f"Checkpoint at {unit} {timestep} saved.")
    
    def __chk(self, epoch:int=None, step:int=None):
        if self.conf.scheme.unit == 'step' and step is not None:
            if step % self.conf.scheme.save_interval == 0:
                if dist.is_initialized():
                    self.learner.opt.consolidate_state_dict(0)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self._save_checkpoint(step, 'step')
                self.log.info(f"Epoch {step} saved")
        elif self.conf.scheme.unit == 'epoch' and epoch is not None:
            if epoch % self.conf.scheme.save_interval == 0:
                if dist.is_initialized():
                    self.learner.opt.consolidate_state_dict(0)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self._save_checkpoint(epoch, 'epoch')
                self.log.info(f"Epoch {epoch} saved")

    @distributed
    def fit(
        self,
        device:int=0,
        global_rank:int=0,
        silent:bool=False,
        position:int=0,
        final:bool=True,
        ):
        
        resumed = self.prepare(device, log=self.log,)

        total_step = 0
        saved = False
        
        loss = None
        s_loss = torch.zeros(2, device=device, requires_grad=False)
        e_loss = torch.zeros(2, device=device, requires_grad=False)

        if resumed:
            self.set_rand_state(resumed['rng_state'])
            self.conf.init_step = resumed['timestep']
            self.conf.unit = resumed['unit']

            if self.conf.unit == 'step':
                total_step = self.conf.init_step
            
            self.log.info("Training state resumed.")

        with Progress(
            *([TextColumn("{task.description}"),
            SpinnerColumn('monkey'),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),])
            ) as prog:

            if self.conf.scheme.unit == "epoch":
                epoch_bar = prog.add_task("Epoch", total=self.conf.scheme.max_step, completed=self.conf.scheme.init_step)
                step_bar = prog.add_task("Update Step", total=len(self.loader))
            else:
                epoch_bar = prog.add_task("Epoch", total=1, visible=False)
                step_bar = prog.add_task("Update Step", total=self.conf.scheme.max_step, completed=self.conf.scheme.init_step)

            for epoch in range(self.conf.scheme.init_step, self.conf.scheme.max_step if self.conf.scheme.unit == 'epoch' else 1):

                if self.conf.scheme.unit == "epoch":
                    prog.reset(step_bar)

                for current_step, batch in enumerate(self.loader):
                    saved = False
                    batch = to_tensor(batch, dtype=self.conf.scheme.dtype, device=device)
                    self.learner.model.train(True)
                    self.learner.opt.zero_grad()

                    if self.conf.scheme.precision is None:
                        loss = self.learner.fit(batch, device, self.log)
                    else:
                        with torch.autocast(torch.device(device), dtype=getattr(torch, self.conf.scheme.precision)):
                            loss = self.learner.fit(batch, device, self.log)

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

                    s_loss[1] = (batch[0].size(dim=0))
                    s_loss[0] = loss.item() * s_loss[1]

                    e_loss[0] += s_loss[0]
                    e_loss[1] += s_loss[1]

                    if dist.is_initialized():
                        dist.reduce(s_loss, 0, dist.ReduceOp.SUM)

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        self.log.log_state({'Training loss' : (s_loss[0] / s_loss[1]).item()}, current_step, 'step', 'Training Loss (Step)')
                    #step_bar.set_postfix({'loss(step)':f'{s_loss[0] / s_loss[1]:.8f}'})

                    total_step += 1
                    prog.update(step_bar, advance=1,)

                    #Evaluation (step)
                    if self.grader is not None:
                        _rng_state = self.get_rand_state()
                        self.grader.eval(learner=self.learner, timestep=total_step, unit='step', final=False, position=position+2, env_conf=self.conf.env)
                        self.set_rand_state(_rng_state)

                    #Checkpoint (step)
                    self.__chk(step=total_step)
                
                # Scheduler
                if self.learner.sch:
                    self.learner.sch.step()

                if dist.is_initialized():
                        dist.reduce(e_loss, 0, dist.ReduceOp.SUM)

                if not dist.is_initialized() or dist.get_rank() == 0:
                    self.log.log_state({'Training loss' : (e_loss[0] / e_loss[1]).item()}, epoch+1, 'epoch', 'Training Loss (Epoch)')

                e_loss[0] = e_loss[1] = 0

                prog.update(epoch_bar, advance=1)

                self.__chk(epoch=epoch+1)

                #Evaluation (epoch)
                if self.grader is not None:
                    _rng_state = self.get_rand_state()
                    self.grader.eval(learner=self.learner, timestep=epoch+1, unit='epoch', final=False, position=position+2, env_conf=self.conf.env)
                    self.set_rand_state(_rng_state)

        # loop over

        if dist.is_initialized():
            self.learner.opt.consolidate_state_dict()
            
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._save_checkpoint(self.conf.scheme.max_step, self.conf.scheme.unit)

        if dist.is_initialized() and final:
            dist.destroy_process_group()