from typing import TypeVar, Literal, Callable
from .grader import TorchGrader
from .base import TorchPredictor, to_tensor, distributed
from ..base import Trainer, Warning
from ..data.course import Course, FullCourse
from .learner import TorchLearner, load_learner
from ..utils.statelogger import StateLogger
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
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

def torchload(cfg:DictConfig, log:L):

    log.info(f"Initiating an experiment based on PyTorch.")

    grader = TorchGrader(cfg.grader)

    # instantiate trainer
    trainer = TorchTrainer(cfg.trainer, cfg.learner, cfg.data, log=log)

    if trainer:
        log.info("Trainer instance created.")
    
    return trainer

class TorchTrainer(Trainer, TorchPredictor):
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

        if 'seed' in conf and conf['seed']:
            self.rand_all(conf.seed)
        else:
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

    def get_loader(self, dataset:Literal['fit', 'eval', 'predict']) -> DataLoader:
        dataset = self.course.get_dataset(dataset)
        if dist.is_initialized():
            return DataLoader(dataset, batch_size=self.conf.batch_size, num_workers=self.conf.num_workers) if not dist.is_initialized() else \
            DataLoader(dataset, batch_size=self.conf.batch_size,  num_workers=self.conf.num_workers, sampler=DistributedSampler(dataset, self.conf.env.world_size, self.conf.env.global_rank))
        else:
            return DataLoader( dataset, num_workers=self.conf.num_workers)

    @distributed
    def fit(self, device:int=0, global_rank:int=0, silent:bool=False, position:int=0):
        
        self.prepare(device, 'fit', self.conf.accel, self.learner_conf, self.data_conf, self.log, self.conf.env.optimizer)

        current_step = 0
        saved = False

        if self.conf.resume:
            if not isinstance(self.conf.resume_step, int):
                raise ValueError("Tried to resume training state, but 'resume_step' is not valid.")

            if not os.path.exists(os.path.abspath(self.conf.resume_dir)):
                raise ValueError("Tried to resume training state, but 'resume_dir' is not valid.")

            model_path = os.path.join(self.conf.resume_dir, f'model_{self.conf.resume_step}.cpt')
            state_path = os.path.join(self.conf.resume_dir, f'state_{self.conf.resume_step}.cpt')

            self.learner.load_model(path=model_path)

            checkpoint = torch.load(state_path)
            self.learner.load_optimizer(state=checkpoint['optimizer'])
            self.set_rand_state(checkpoint['rng_state'])
            self.conf.init_step = checkpoint['timestep']
            self.conf.unit = checkpoint['unit']

            if self.conf.unit == 'step':
                current_step = self.conf.init_step

        with logging_redirect_tqdm():
            for epoch in (epoch_bar:=tqdm(range(self.conf.init_step, self.conf.max_step if self.conf.unit == 'epoch' else 1), desc='Epoch', unit='epoch', position=0, leave=False, disable=True if self.conf.unit != 'epoch' else silent)):
                for batch in (step_bar:=tqdm(self.loader if self.conf.unit == 'epoch' else range(self.conf.init_step, self.conf.max_step), desc='Steps', unit='step', position=1, leave=False, disable=silent)):

                    saved = False

                    batch = to_tensor(batch)
                    self.learner.model.train(True)
                    pred = self.learner.fit(batch, device, self.log)
                    tr_loss = self.learner.loss_fn(pred, batch[1].to(device))
                    self.learner.opt.zero_grad()
                    tr_loss.backward()
                    self.learner.opt.step()
                    self.learner.model.train(False)

                    step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step+1}
                    self.log.log_state(step_postfix)
                    step_postfix['training_loss'] = '{:.5f}'.format(step_postfix['training_loss'])
                    step_bar.set_postfix(step_postfix)

                    current_step += 1

                    if self.conf.unit == 'step':
                        if current_step % self.conf.save_interval == 0:
                            if dist.is_initialized():
                                self.learner.opt.consolidate_state_dict()
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                self._save_checkpoint(current_step, 'step')
                                saved = True

                if self.learner.sch:
                    self.learner.sch.step()

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

        if dist.is_initialized():
            dist.destroy_process_group()