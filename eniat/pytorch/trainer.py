from typing import TypeVar, Literal, Callable
from ..base import Trainer, Warning
from ..data.course import Course, FullCourse
from .learner import TorchLearner
from tqdm.auto import tqdm
from tqdm.contrib import DummyTqdmFile
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.multiprocessing import spawn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer
import torch.distributed.algorithms.model_averaging.averagers as averagers
import sys
import os
import random
import numpy as np
from omegaconf import DictConfig
from multiprocessing import current_process
from ..data.course import get_course_instance, batch_load
from .._dyn import conf_instantiate, _dynamic_import
from hydra.utils import instantiate
from importlib import import_module
from contextlib import contextmanager
import warnings

T = TypeVar('T', bound=TorchLearner)
C = TypeVar('C', bound=FullCourse)

@contextmanager
def _stdout():
    systream = sys.stdout, sys.stderr
    # IO (Main process)
    if (_pname:=current_process().name) != "SpawnProcess-1" and _pname != "MainProcess":
        try:
            with open(os.devnull, 'w') as f:
                sys.stdout, sys.stderr = f, f
                yield sys.stdout
        except Exception as e:
            raise e
        finally:
            sys.stdout, sys.stderr = systream
    else:
        yield systream[0]

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
        torch.save(self.learner.model.state_dict(), os.path.join(os.getcwd(), f'checkpoints/model_{timestep}.cpt' if timestep else 'checkpoints/model.cpt'))

    def _save_train_state(self, timestep:int) -> None:
        train_state = self.learner.get_optimizer().state_dict()['optimizer']
        train_state['unit'] = self.unit
        train_state['rng_state'] = self.get_rand_state()
        train_state['timestep'] = timestep
        train_state['maxstep'] = self.max_step
        train_state['distributed'] = self._dist
        torch.save(train_state, os.path.join(os.getcwd(), f'checkpoints/state_{timestep}.cpt'))

    def _save_checkpoint(self, timestep:int, unit:Literal['epoch', 'step'], force:bool=False) -> None:
        if not force and (self.conf.unit != unit or (timestep % self.conf.save_interval) if self.conf.save_interval else True):
            return
        self._save_model(timestep)
        self._save_train_state(timestep)
        self.log.info(f"Checkpoint at {timestep} {unit} saved.")

    def get_loader(self, dataset:Literal['fit', 'eval', 'predict']) -> DataLoader:
        return DataLoader(self.course.get_dataset(dataset), num_workers=self.conf.num_workers)

    def fit(self, device:int, silent:bool=False):
        current_step = 0
        self.info("test")
        with _stdout() as stdout:
            for epoch in (epoch_bar:=tqdm(range(self.init_step, self.max_step if self.unit == 'epoch' else 1), desc='Training', unit='epoch', position=0, leave=False, disable=True if self.unit != 'epoch' else silent, file=stdout, dynamic_ncols=True)):
                for batch in (step_bar:=tqdm(self.loader, desc='Batch', unit='step', position=1, leave=False, disable=silent, _file=_stdout)):
                    batch = self.to_tensor(batch)
                    tr_loss = self.learner.fit(batch, device, self.log)
                    self.learner.opt.zero_grad()
                    tr_loss.backward()
                    self.learner.opt.step()
                    step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step}
                    step_bar.set_postfix(step_postfix)
                    # Step Log
                    self.log.log_state(step_postfix)
                    # Step Eval
                    # Step Save
                    if self.unit == 'step' and current_step % self.conf.save_interval == 0:
                        self._save_checkpoint(current_step, 'step')
                    current_step += 1
                self.learner.sch.step()
                self.log.info(f"Epoch {epoch} finished")

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

    def eval(self, device:int, silent:bool=False):
        current_step = 0
        for epoch in (epoch_bar:=tqdm(range(self.init_step, self.max_step if self.unit == 'epoch' else 1), desc='Training', unit='epoch', position=0, leave=False, disable=True if self.unit != 'epoch' else silent)):
            for batch in (step_bar:=tqdm(self.loader, desc='Batch', unit='step', position=1, leave=False, disable=silent)):
                batch = self.to_tensor(batch)
                tr_loss = self.learner.fit(batch, device, self.log)
                self.learner.opt.zero_grad()
                tr_loss.backward()
                self.learner.opt.step()
                step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step}
                step_bar.set_postfix(step_postfix)
                # Step Log
                self.log.log_state(step_postfix)
                # Step Eval
                # Step Save
                if self.unit == 'step' and current_step % self.conf.save_interval == 0:
                    self._save_checkpoint(current_step, 'step')
                current_step += 1
            self.learner.epoch()
            self.log.info(f"Epoch {epoch} finished")

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

    def predict(self, device:int, silent:bool=False):
        current_step = 0
        for epoch in (epoch_bar:=tqdm(range(self.init_step, self.max_step if self.unit == 'epoch' else 1), desc='Training', unit='epoch', position=0, leave=False, disable=True if self.unit != 'epoch' else silent)):
            for batch in (step_bar:=tqdm(self.loader, desc='Batch', unit='step', position=1, leave=False, disable=silent)):
                batch = self.to_tensor(batch)
                tr_loss = self.learner.fit(batch, device, self.log)
                self.learner.opt.zero_grad()
                tr_loss.backward()
                self.learner.opt.step()
                step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step}
                step_bar.set_postfix(step_postfix)
                # Step Log
                self.log.log_state(step_postfix)
                # Step Eval
                #self.grader.compute()
                # Step Save
                if self.unit == 'step' and current_step % self.conf.save_interval == 0:
                    self._save_checkpoint(current_step, 'step')
                current_step += 1
            self.learner.epoch()
            self.log.info(f"Epoch {epoch} finished")
            #self.log.log_state()

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

class TorchDistributedTrainer(TorchTrainer):
    r"""PyTorch compatible trainer class, and supports distributed learning (DistirbutedDataParallel aka DDP or Torchrun).
    Because Eniat basically load necessary components and data dynamically, TorchTrainer does not work in the distributed environment.
    TorchDistributedTrainer is technically doing sames tasks as TorchTrainer, but it loads components after spawned processes started. Because of that, it receives only config parameters, and handles loading by itself."""
    def __init__(self, conf:DictConfig=None, learner_conf:DictConfig=None, data_conf:DictConfig=None, grader=None, logger=None) -> None:

        if not conf.distributed or conf.distributed.type == 'none':
            self._dist = False
        else: # distributed learning set
            self._dist = True

        self.conf = conf
        self.learner_conf = learner_conf
        self.data_conf = data_conf

        self.log = logger

        self.seed = conf.seed
        self.unit = conf.unit
        self.save_interval = conf.save_interval
        self.compile = conf.accel
        self.init_step = conf.init_step
        self.max_step = conf.max_step
        
    def dist_init(self, local_rank:int, fname:str) -> None:
        self.log.debug("Spawn entry entered")
        if self.conf.distributed.debug:
            os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.conf.distributed.type == "torchrun":
            self.log.info("setting torchrun environment...")
            local_size = self.conf.distributed.local_size = int(os.environ['LOCAL_WORLD_SIZE'])
            self.conf.distributed.local_rank = int(os.environ['LOCAL_RANK'])
            self.conf.distributed.global_rank = int(os.environ['RANK'])
            self.conf.distributed.world_size = int(os.environ['WORLD_SIZE'])
            init_process_group(backend=self.conf.distributed.backend)
            getattr(self, fname)(self.conf.distributed.local_rank)
        elif self.conf.distributed.type == "DDP":
            self.log.info("setting DDP environment...")
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["MASTER_ADDR"] = self.conf.distributed.master_address
            os.environ["MASTER_PORT"] = self.conf.distributed.master_port
            rank = self.conf.distributed.global_rank + local_rank
            init_process_group(backend=self.conf.distributed.backend, world_size=self.conf.distributed.world_size, rank=rank)
            return getattr(self, fname)(local_rank)
        else:
            raise ValueError("The type of distributed config must be one of the following literals: ['torchrun', 'DDP', 'none']")

    def distributed (fn:Callable) -> Callable:
        def wrapper(self, *args):
            self.log.debug(current_process().name)
            if current_process().name == "MainProcess":
                warnings.showwarning = Warning(self.log)
                if self._dist:
                    if self.conf.distributed.type == "DDP":
                        if current_process().name == "MainProcess":
                            spawn(self.dist_init, (fn.__name__,), nprocs=self.conf.distributed.local_size, join=True)
                    elif self.conf.distirbuted.type == "torchrun":
                        return self.dist_init(self, int(os.environ['LOCAL_RANK']), fn.__name__)
                else:
                    return fn(self, *args)
            else:
                if current_process().name == "SpawnProcess-1":
                    self.log.info("Custom Warning")
                    warnings.showwarning = Warning(self.log)
                else:
                    self.log.info("warning ignored")
                    warnings.filterwarnings("ignore")
                return fn(self, *args)
        return wrapper

    def prepare (self, device:int, task:Literal['fit', 'eval', 'predict']):
        # data
        self.course = FullCourse()
        cfg = self.data_conf
        for label in cfg:
            if 'cls' in cfg[label]:
                self.course.append(Course(label, conf_instantiate(cfg[label])))
                self.log.info(f"'{label}' data is loaded")
            elif 'path' in cfg[label]:
                self.course.append(course=Course(label, data=batch_load(cfg[label]['path'], cfg[label].type)))
                self.log.info(f"'{label}' data is loaded.")
            else:
                self.log.warning(f"Data(:{label}) is not loaded because the path of data is not specified.")
        if not len(self.course):
            self.log.warning("No data is given! Terminating the task...")
            return

        self.course = get_course_instance(self.data_conf, self.log)
        self.loader = self.get_loader(task)

        # learner
        cfg = self.learner_conf
        model = conf_instantiate(cfg.model)
        self.log.info("Model loaded...")

        if not cfg.resume:
            self.log.warning("'resume' is set to False. The model will be initialized without loading a checkpoint.")
        # loss
        loss = instantiate(cfg.loss) if cfg.loss and cfg.loss._target_ else None
        if loss:
            self.log.info("Loss function loaded...")
        else:
            self.log.warning("Loss function is not defined. Are you sure you wanted this?")
        # optimizer
        optim = instantiate(cfg.optimizer, params=model.parameters()) if cfg.optimizer and cfg.optimizer._target_ else None
        if optim:
            self.log.info("Optimizer loaded...")
        else:
            self.log.warning("Optimizer is not defined. Are you sure you wanted this?")
        # scheduler
        schlr = instantiate(cfg.scheduler, lr_lambda=lambda x: x**cfg.scheduler.lr_lambda, optimizer=optim) if cfg.scheduler and cfg.scheduler._target_ else None
        if schlr:
            self.log.info("Scheduler loaded...")
        else:
            self.log.warning("Scheduler is not defined. Edit the configuration if this is not what you wanted.")
        
        if self._dist:
            model = DDP(model.to(device)).compile() if self.compile else DDP(model)
            optim = self.get_dist_opt(self.conf.distributed.optimizer, optim, model.parameters())
        
        # instantiate learner
        if 'path' in cfg and cfg.path:
            _mod, _bn = _dynamic_import(cfg.path)
            self.learner = getattr(_mod, cfg.cls)(model, loss, optim, schlr, cfg.resume, cfg.resume_path)
        else:
            self.learner = getattr(import_module('.pytorch.learner', 'eniat'), cfg.cls)(model, loss, optim, schlr, cfg.resume, cfg.resume_path)
        if self.learner:
            self.log.info("Learner instance created.")
    
    @staticmethod
    def to_tensor(batch):
        if isinstance(batch, list):
            for data in batch:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(data)
        return batch

    @distributed
    def fit(self, device:int=0, global_rank:int=None, silent:bool=False, init_timestemp:int=0):
        silent = True if device != 0 else silent
        self.prepare(device, 'fit')
        current_step = 0

        for epoch in (epoch_bar:=tqdm(range(self.init_step, self.max_step if self.unit == 'epoch' else 1), desc='Mini Batch   ', unit='epoch', position=0, leave=False, disable=True if self.unit != 'epoch' else silent)):
            for batch in (step_bar:=tqdm(self.loader, desc='Training step', unit='step', position=1, leave=False, disable=silent)):
                batch = self.to_tensor(batch)
                tr_loss = self.learner.fit(batch, device, self.log)
                self.learner.opt.zero_grad()
                tr_loss.backward()
                self.learner.opt.step()
                step_postfix = {'training_loss' : tr_loss.item(), 'step': current_step}
                step_bar.set_postfix(step_postfix)
                # Step Log
                self.log.log_state(step_postfix)
                # Step Eval

                # Step Save
                if self.unit == 'step' and current_step % self.conf.save_interval == 0:
                    self._save_checkpoint(current_step, 'step')
                current_step += 1
            self.learner.sch.step()
            self.log.info(f"Epoch {epoch} finished")
            #self.log.log_state()

        self._save_checkpoint(self.conf.max_step, self.conf.unit)

        if self.distributed:
            destroy_process_group()

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

        if final and self.distributed:
            destroy_process_group()

        if self.eval_fn:
            return self.eval_fn(whole_batch)
        else:
            return whole_batch

    @distributed
    def predict(self, device:int, final:bool=False, silent:bool=False):

        silent = device != 0 and silent
        self.prepare(device, 'predict')
        # Batch Inference
        outputs = None
        for batch in tqdm(self.loader, unit='Steps', position=1, disable=silent):
            output = self.learner.predict(batch)
            if not outputs:
                outputs = torch.empty((0, *output.shape[1:]))
            outputs = torch.cat((outputs, output), dim=0)

        if final and self.distributed:
            destroy_process_group()

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