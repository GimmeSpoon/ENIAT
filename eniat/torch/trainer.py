from typing import TypeVar, Literal, Callable
from ..base import Trainer, Warning
from ..data.course import Course, FullCourse
from .learner import TorchLearner, load_learner
from ..utils.statelogger import StateLogger
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer
import torch.distributed.algorithms.model_averaging.averagers as averagers
import os
import random
import numpy as np
from omegaconf import DictConfig
from ..data.course import get_course_instance
from importlib import import_module
import warnings
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from pathlib import Path

T = TypeVar('T', bound=TorchLearner)
C = TypeVar('C', bound=FullCourse)
L = TypeVar('L', bound=StateLogger)

def torchload(cfg:DictConfig, log:L):

    log.info(f"Initiating an experiment based on PyTorch.")

    if cfg.trainer.distributed.type == "none" or cfg.trainer.distributed.type == "DP":
    
    # DATA LOAD
        _courses = get_course_instance(cfg.data, log)
        log.info('Loaded dataset.\n' + _courses.__repr__())
        
        # instantiate learner components
        learner = load_learner(cfg.learner, log)

        # instantiate trainer
        trainer = getattr(import_module('.pytorch', 'eniat'), 'TorchTrainer')(course=_courses, learner=learner, conf=cfg.trainer, logger=log)
    
    if trainer:
        log.info("Trainer instance created.")
    
    return learner, trainer

class TorchTrainer(Trainer):
    r"""PyTorch compatible trainer class.
    Automatically manage trainign step, logging, and saving checkpoints. Takse one task, one dataset, and one learner for any run. For several tasks, you can initiate the same number of Trainers."""
    def __init__(self, course: C = None, learner: T = None, conf=None, grader=None, logger=None) -> None:
        super().__init__(course, learner, conf, grader, logger)
        warnings.showwarning = Warning(self.log)

    def rand_all(self, seed):
        if self._dist:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def get_rand_state(self) -> dict:
        return {
            'cuda' : torch.cuda.get_rng_state_all() if self._dist else torch.cuda.get_rng_state(),
            'torch' : torch.get_rng_state(),
            'numpy' : np.random.state(),
            'random' : random.getstate()
        }
    
    def set_rand_state(self, state:dict) -> None:
        if self._dist:
            torch.cuda.set_rng_state_all(state['cuda'])
        else:
            torch.cuda.set_rng_state(state['cuda'])
        torch.set_rng_state(state['torch'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.set_state(state['numpy'])
        random.setstate(state['random'])

    def resume_training_state(self, path:str) -> None:
        resumed = torch.load(path)
        self.unit = resumed['unit']
        self.set_rand_state(resumed['rng_state'])
        self.init_step = resumed['timestep']
        self.max_step = resumed['maxstep']

    def _save_model(self, timestep:int=None) -> None:
        if not self.hc:
            self.hc = HydraConfig.get()
        Path(os.path.join(self.hc.runtime.output_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
        torch.save(self.learner.model.state_dict(), os.path.join( self.hc.runtime.output_dir , f'checkpoints/model_{timestep}.cpt' if timestep else 'checkpoints/model.cpt'))

    def _save_train_state(self, timestep:int) -> None:
        if not self.hc:
            self.hc = HydraConfig.get()
        train_state = self.learner.get_optimizer().state_dict()['optimizer']
        train_state['unit'] = self.unit
        train_state['rng_state'] = self.get_rand_state()
        train_state['timestep'] = timestep
        train_state['maxstep'] = self.max_step
        train_state['distributed'] = self._dist
        Path(os.path.join(self.hc.runtime.output_dir, 'checkoints')).mkdir(parents=True, exist_ok=True)
        torch.save(train_state, os.path.join(self.hc.runtime.output_dir, f'checkpoints/state_{timestep}.cpt'))

    def _save_checkpoint(self, timestep:int, unit:Literal['epoch', 'step'], force:bool=False) -> None:
        if not force and (self.conf.unit != unit or (timestep % self.conf.save_interval) if self.conf.save_interval else True):
            return
        self._save_model(timestep)
        self._save_train_state(timestep)
        self.log.info(f"Checkpoint at {timestep} {unit} saved.")

    def get_loader(self, dataset:Literal['fit', 'eval', 'predict']) -> DataLoader:
        return DataLoader(self.course.get_dataset(dataset), num_workers=self.conf.num_workers)

    @staticmethod
    def to_tensor(batch):
        if isinstance(batch, list):
            for data in batch:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(data)
        return batch

    def fit(self, device:int=0, silent:bool=False):
        loader = self.get_loader('fit')
        current_step = 0
        with logging_redirect_tqdm():
            for epoch in (epoch_bar:=tqdm(range(self.conf.init_step, self.conf.max_step if self.conf.unit == 'epoch' else 1), desc='Epoch', unit='epoch', position=0, leave=False, disable=True if self.conf.unit != 'epoch' else silent)):
                for batch in (step_bar:=tqdm(loader, desc='Steps', unit='step', position=1, leave=False, disable=silent)):
                    batch = self.to_tensor(batch)
                    self.learner.model.train(True)
                    tr_loss = self.learner.fit(batch, device, self.log)
                    self.learner.opt.zero_grad()
                    tr_loss.backward()
                    self.learner.opt.step()
                    self.learner.model.train(False)
                    step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step+1}
                    self.log.log_state(step_postfix)
                    step_postfix['trainintorg_loss'] = '{:.5f}'.format(step_postfix['training_loss'])
                    step_bar.set_postfix(step_postfix)
                    if self.conf.unit == 'step' and current_step % self.conf.save_interval == 0:
                        self._save_checkpoint(current_step, 'step')
                    current_step += 1
                if self.learner.sch:
                    self.learner.sch.step()
                if self.conf.unit == 'epoch':
                    self.log.info(f"Epoch {epoch+1} finished")

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

    def eval(self, device:int=0, silent:bool=False):
        loader = self.get_loader('eval')

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

    def predict(self, device:int=0, silent:bool=False):
        loader = self.get_loader('predict')
        current_step = 0
        for batch in (step_bar:=tqdm(loader, desc='Steps', unit='step', position=1, leave=False, disable=silent)):
            batch = self.to_tensor(batch)
            tr_loss = self.learner.fit(batch, device, self.log)
            self.learner.opt.zero_grad()
            tr_loss.backward()
            self.learner.opt.step()
            step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step}
            step_bar.set_postfix(step_postfix)
            self.log.log_state(step_postfix)
            if self.unit == 'step' and current_step % self.conf.save_interval == 0:
                self._save_checkpoint(current_step, 'step')
            current_step += 1

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

class TorchDistributedTrainer(TorchTrainer):
    r"""PyTorch compatible trainer class, and supports distributed learning (DistirbutedDataParallel aka DDP or Torchrun).
    Because Eniat basically load necessary components and data dynamically, TorchTrainer does not work in the distributed environment.
    TorchDistributedTrainer is technically doing sames tasks as TorchTrainer, but it loads components after spawned processes started. Because of that, it receives only config parameters, and handles loading by itself."""
    def __init__(self, conf:DictConfig=None, learner_conf:DictConfig=None, data_conf:DictConfig=None, logger_conf:DictConfig=None, grader=None) -> None:

        if not conf.distributed or conf.distributed.type == 'none':
            self._dist = False
        else: # distributed learning set
            self._dist = True

        self.conf = conf
        self.learner_conf = learner_conf
        self.data_conf = data_conf

        self.log = StateLogger('eniat', conf=logger_conf)

        self.seed = conf.seed
        self.unit = conf.unit
        self.save_interval = conf.save_interval
        self.compile = conf.accel
        self.init_step = conf.init_step
        self.max_step = conf.max_step
        
    def _ddp_init(self, local_rank:int, fname:str, _hc=None) -> None:
        configure_log(_hc.job_logging, _hc.verbose)
        self.hc = _hc
        self.log.info("setting DDP environment...")
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = self.conf.distributed.master_address
        os.environ["MASTER_PORT"] = self.conf.distributed.master_port
        rank = self.conf.distributed.global_rank + local_rank
        dist.init_process_group(backend=self.conf.distributed.backend, world_size=self.conf.distributed.world_size, rank=rank)
        return getattr(self, fname)(local_rank, rank)

    def _torchrun_init(self):
        self.log.info("setting torchrun environment...")
        self.conf.distributed.local_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self.conf.distributed.local_rank = int(os.environ['LOCAL_RANK'])
        self.conf.distributed.global_rank = int(os.environ['RANK'])
        self.conf.distributed.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend=self.conf.distributed.backend)

    def distributed (fn:Callable) -> Callable:
        def wrapper(self, *args):
            if not dist.is_initialized():
                if self.conf.distributed.debug:
                    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
                    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
                warnings.showwarning = Warning(self.log)
                if dist.is_torchelastic_launched():
                    self.hc = HydraConfig.get()
                    self._torchrun_init(self)
                    return fn(int(os.environ['LOCAL_RANK']), int(os.environ['RANK']))
                else:
                    spawn(self._ddp_init, (fn.__name__, HydraConfig.get()), nprocs=self.conf.distributed.local_size, join=True)
            else:
                if dist.get_rank() == 0:
                    warnings.showwarning = Warning(self.log)
                    return fn(self, *args)
                else:
                    warnings.filterwarnings("ignore")
                    with self.log.silent():
                        return fn(self, *args)
        return wrapper

    def prepare (self, device:int, task:Literal['fit', 'eval', 'predict']):
        # data
        self.course = get_course_instance(self.data_conf, self.log)
        self.loader = self.get_loader(task)

        # learner
        self.learner = load_learner(self.learner_conf, self.log)
        model = self.learner.model

        if self._dist:
            model = DDP(model.to(device)).compile() if self.compile else DDP(model)
            optim = self.get_dist_opt(self.conf.distributed.optimizer, self.learner.opt, model.parameters())

    @distributed
    def fit(self, device:int=0, global_rank:int=0, silent:bool=False):
        silent = True if device != 0 else silent
        self.prepare(device, 'fit')
        current_step = 0

        with logging_redirect_tqdm():
            for epoch in (epoch_bar:=tqdm(range(self.init_step, self.max_step if self.unit == 'epoch' else 1), desc='Epoch', unit='epoch', position=0, leave=False, disable=True if self.unit != 'epoch' else silent)):
                for batch in (step_bar:=tqdm(self.loader, desc='Steps', unit='step', position=1, leave=False, disable=silent)):
                    batch = self.to_tensor(batch)
                    tr_loss = self.learner.fit(batch, device, self.log)
                    self.learner.opt.zero_grad()
                    tr_loss.backward()
                    self.learner.opt.step()
                    step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step}
                    self.log.log_state(step_postfix)
                    step_postfix['training_loss'] = '{:.5f}'.format(step_postfix['training_loss'])
                    step_bar.set_postfix(step_postfix)
                    if self.unit == 'step' and current_step % self.conf.save_interval == 0 and global_rank == 0:
                        self._save_checkpoint(current_step, 'step')
                    current_step += 1
                if self.learner.sch:
                    self.learner.sch.step()
                if self.conf.unit == 'epoch':
                    self.log.info(f"Epoch {epoch+1} finished")

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

        if dist.is_initialized():
            dist.destroy_process_group()

    @distributed
    def eval(self, device:int, final:bool=False, silent:bool=False):

        silent = device != 0 and silent
        self.prepare(device, 'eval')
        # Evaluation
        if not self.course:
            raise AttributeError("No evaluation dataset.")
            
        whole_batch = None
        for batch in tqdm(self.course, desc='Validation', unit='step', leave=False, disable=silent):
            output = self.learner.infer(batch, device)
            if not whole_batch:
                whole_batch = torch.empty((0, *output.shape[1:]))
            whole_batch = torch.cat((whole_batch, output), dim=0)

        if final and dist.is_initialized():
            dist.destroy_process_group()

        if self.eval_fn:
            return self.eval_fn(whole_batch)
        else:
            return whole_batch

    @distributed
    def predict(self, device:int, final:bool=False, silent:bool=False):

        silent = device != 0 and silent
        self.prepare(device, 'predict')
        # Steps Inference
        outputs = None
        for batch in tqdm(self.loader, unit='step', position=1, disable=silent):
            output = self.learner.predict(batch)
            if not outputs:
                outputs = torch.empty((0, *output.shape[1:]))
            outputs = torch.cat((outputs, output), dim=0)

        if final and dist.is_initialized():
            dist.destroy_process_group()

        return outputs

    def get_dist_opt(self, method:Literal['zero', 'postlocal'], opt=None, params=None, **kwargs):
        if method == 'zero':
            if not params:
                raise ValueError("model parameters require for zero redundancy optimizer.")
            return ZeroRedundancyOptimizer(params, optimizer_class=getattr(torch.optim, type(opt).__name__), parameters_as_bucket_view=False, overlap_with_ddp=False, **kwargs)
        if method == 'postlocal':
            return PostLocalSGDOptimizer(opt, averagers.PeriodicModelAverager(**kwargs))
        
    def get_loader(self, dataset:Literal['fit', 'eval', 'predict']) -> DataLoader:
        dataset = self.course.get_dataset(dataset)
        return DataLoader(dataset, batch_size=self.conf.batch_size, num_workers=self.conf.num_workers) if not self._dist else \
        DataLoader(dataset, batch_size=self.conf.batch_size,  num_workers=self.conf.num_workers, sampler=DistributedSampler(dataset, self.conf.distributed.world_size, self.conf.distributed.global_rank))