from typing import TypeVar, Literal, Callable
from .grader import TorchGrader
from .base import TorchPredictor, to_tensor, distributed
from ..base import Trainer, Warning
from ..data.course import Course, FullCourse
from .learner import TorchLearner
from ..utils.statelogger import StateLogger
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
import os
import random
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import warnings

T = TypeVar('T', bound=TorchLearner)
C = TypeVar('C', bound=FullCourse)
L = TypeVar('L', bound=StateLogger)
G = TypeVar('G', bound=TorchGrader)

def torchload(cfg:DictConfig, log:L):

    log.info(f"Initiating an experiment based on PyTorch.")

    grader = TorchGrader(cfg.grader, logger=log)
    # instantiate trainer
    trainer = TorchTrainer(cfg.trainer, cfg.learner, cfg.data, log=log, grader=grader if grader.is_enabled() else None)

    if trainer:
        log.info("Trainer instance created.")
    
    return trainer, grader

class TorchTrainer(TorchPredictor, Trainer):
    r"""PyTorch compatible trainer class.
    Automatically manage training steps, logging, and saving checkpoints.
    It only occupies one device(GPU)"""
    def __init__(self, conf:DictConfig=None, learner_conf:DictConfig=None, data_conf:DictConfig=None, logger_conf:DictConfig=None, log=None, grader=None) -> None:

        self.conf = conf
        self.learner_conf = learner_conf
        self.data_conf = data_conf

        if log is not None:
            self.log = log
        else:
            self.log = StateLogger('eniat', conf=logger_conf)

        self.grader = grader

        self.seed = conf.seed
        self.unit = conf.unit
        self.save_interval = conf.save_interval
        self.compile = conf.accel
        self.init_step = conf.init_step
        self.max_step = conf.max_step
        self.batch_size = conf.batch_size

        self.hc = HydraConfig.get()

        warnings.showwarning = lambda *args: self.log.warning(args[0])

        self.hooks = {
            'f': [], 'b': [], 'bs': [], 'as': [], 'be': [],'ae': []
        }

        if conf.seed and not conf.resume_opt:
            self.rand_all(conf.seed)
        elif not conf.seed:
                self.log.warning("Random seed is not set in the configuration. If there's no resumed state, random behaviors can not be controlled.")
                self.seed = random.random()
                self.rand_all(self.seed)

    def rand_all(self, seed) -> None:
        if self.conf.env.type != 'single':
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        self.log.info(f"Random seed set:{seed}")

    def get_rand_state(self) -> dict:
        return {
            'cuda' : torch.cuda.get_rng_state_all() if self.conf.env.type != 'single' else torch.cuda.get_rng_state(),
            'torch' : torch.get_rng_state(),
            'numpy' : np.random.get_state(),
            'random' : random.getstate()
        }
    
    def set_rand_state(self, state:dict) -> None:
        if self.conf.env.type != 'single':
            torch.cuda.set_rng_state_all(state['cuda'])
            for i, rng_state in enumerate(state['cuda']):
                torch.cuda.set_rng_state(rng_state)
        else:
            torch.cuda.set_rng_state(state['cuda'])
        torch.set_rng_state(state['torch'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.set_state(state['numpy'])
        random.setstate(state['random'])
        self.log.info(f"Random state set.")

    def load_state(self, path:str) -> None:
        resumed = torch.load(path)
        self.learner.load_optimizer(resumed['optimizer'])
        self.unit = resumed['unit']
        self.set_rand_state(resumed['rng_state'])
        self.init_step = resumed['timestep']
        self.batch_size = resumed['batch_size']
        self.max_step = resumed['maxstep']
        self.conf.env = resumed['env']
        self.log.info("Random state loaded.")

    def _save_model(self, timestep:int=None) -> None:
        if not self.hc:
            self.hc = HydraConfig.get()
        Path(os.path.join(self.hc.runtime.output_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(self.learner.model.state_dict(), os.path.join( self.hc.runtime.output_dir , f'checkpoints/model_{timestep}.cpt' if timestep else 'checkpoints/model.cpt'))

    def _save_state(self, timestep:int, filename:str=None) -> None:
        if not self.hc:
            self.hc = HydraConfig.get()
        train_state = {}
        train_state['optimizer'] = self.learner.opt.state_dict()
        train_state['unit'] = self.unit
        train_state['rng_state'] = self.get_rand_state()
        train_state['timestep'] = timestep
        train_state['batch_size'] = self.batch_size
        train_state['maxstep'] = self.max_step
        train_state['env'] = self.conf.env
        Path(os.path.join(self.hc.runtime.output_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(train_state, os.path.join(self.hc.runtime.output_dir, f'checkpoints/state_{timestep}.cpt' if filename is None else filename))

    def _save_checkpoint(self, timestep:int, unit:Literal['epoch', 'step'], force:bool=False) -> None:
        if not force and (self.conf.unit != unit or (timestep % self.conf.save_interval) if self.conf.save_interval else True):
            return
        self._save_model(timestep)
        self._save_state(timestep)
        self.log.info(f"Checkpoint at {unit} {timestep} saved.")

    def hook(self, when:Literal['f', 'b', 'bs', 'as', 'be', 'ae'], fn:Callable) -> None:
        self.hooks[when].append(fn)

    def get_hooks(self, when:Literal['f', 'b', 'bs', 'as', 'be', 'ae']) -> list[Callable]:
        return self.hooks[when]
    
    def _take_hook(self, when:Literal['f', 'b', 'bs', 'as', 'be', 'ae'], *args) -> None:
        for fn in self.hooks[when]:
            fn(*args)

    @distributed
    def fit(self, device:int=0, global_rank:int=0, silent:bool=False, position:int=0, final:bool=True, data_label:str='fit'):
        
        resumed = self.prepare(device=device, data_label=data_label, compile=self.conf.accel, learner_cfg=self.learner_conf, data_cfg=self.data_conf, log=self.log, dist_opt=self.conf.env.optimizer, resume_model=self.conf.resume_model, resume_opt=self.conf.resume_opt, resume_dir=self.conf.resume_dir, resume_step=self.conf.resume_step)

        current_step = 0
        saved = False
        avg_loss = None
        loss = None

        if resumed:
            self.set_rand_state(resumed['rng_state'])
            self.conf.init_step = resumed['timestep']
            self.conf.unit = resumed['unit']

            if self.conf.unit == 'step':
                current_step = self.conf.init_step

        with logging_redirect_tqdm():
            for epoch in (epoch_bar:=tqdm(range(self.conf.init_step, self.conf.max_step if self.conf.unit == 'epoch' else 1), desc='Epoch', unit='epoch', position=position, leave=False, disable=True if self.conf.unit != 'epoch' else silent)):
                # Before epoch hook
                self._take_hook('be', epoch+1, avg_loss, self.learner)
                avg_loss = whole_batch = 0
                for batch in (step_bar:=tqdm(self.loader if self.conf.unit == 'epoch' else range(self.conf.init_step, self.conf.max_step), desc='Steps', unit='step', position=position+1, leave=False, disable=silent)):
                    saved = False
                    batch = to_tensor(batch)
                    self.learner.model.train(True)
                    self.learner.opt.zero_grad()
                    # Before step hook
                    self._take_hook('bs', current_step+1, loss, self.learner, self)

                    # forward
                    if self.conf.precision is None:
                        loss = self.learner.fit(batch, device, self.log)
                    else:
                        with torch.autocast(torch.device(device), dtype=getattr(torch, self.conf.precision)):
                            loss = self.learner.fit(batch, device, self.log)

                    # forward hook
                    self._take_hook('f', current_step+1, loss, self.learner, self)

                    # backward & optimizer step
                    if self.conf.gradient_scale:
                        scaler = GradScaler(loss)
                        scaler.scale(loss).backward()
                        self._take_hook('b', current_step+1, loss, self.learner, self)
                        scaler.step(self.learner.opt)
                        scaler.update()
                    else:
                        scaler = None
                        loss.backward()
                        self._take_hook('b', current_step+1, loss, self.learner, self)
                        self.learner.opt.step()

                    # After step hook
                    self._take_hook('as', current_step+1, loss, self.learner, self)

                    self.learner.model.train(False)

                    step_postfix = {'training_loss' : loss.item(), 'step': current_step+1}
                    self.log.log_state(step_postfix)
                    step_postfix['training_loss'] = '{:.8f}'.format(step_postfix['training_loss'])
                    step_bar.set_postfix({'loss':step_postfix['training_loss']})

                    avg_loss += loss.item() * (_cbs:=batch[0].size(dim=0))
                    whole_batch += _cbs

                    current_step += 1

                    #Evaluation (step)
                    if self.grader is not None:
                        self.grader.eval(self.learner, self, self.course.get_dataset('eval'), current_step, 'step', step_check=True, final=False, position=position+2)

                    #Checkpoint (step)
                    if self.conf.unit == 'step':
                        if current_step % self.conf.save_interval == 0:
                            if dist.is_initialized():
                                self.learner.opt.consolidate_state_dict()
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                self._save_checkpoint(current_step, 'step')
                                saved = True
                
                # Scheduler
                if self.learner.sch:
                    self.learner.sch.step()

                epoch_postfix = {'avg_loss' : avg_loss / whole_batch, 'step': epoch+1}
                self.log.log_state(epoch_postfix)
                epoch_postfix['avg_loss'] = '{:.8f}'.format(epoch_postfix['avg_loss'])
                epoch_bar.set_postfix({'avg_loss':epoch_postfix['avg_loss']})

                # After epoch hook
                self._take_hook('ae', epoch+1, avg_loss, self.learner, self)

                if self.conf.unit == 'epoch':
                    self.log.info(f"Epoch {epoch+1} finished")
                    if current_step % self.conf.save_interval == 0:
                        if dist.is_initialized():
                                self.learner.opt.consolidate_state_dict()
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            self._save_checkpoint(epoch+1, 'epoch')
                            saved = True

        if dist.is_initialized():
            self.learner.opt.consolidate_state_dict()
            
        if not saved and (not dist.is_initialized() or dist.get_rank() == 0):
            self._save_checkpoint(self.conf.max_step, self.conf.unit)

        if dist.is_initialized() and final:
            dist.destroy_process_group()