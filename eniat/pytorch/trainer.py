from typing import TypeVar, Literal, Callable
from ..base import Trainer
from ..data.course import Course, FullCourse
from .learner import TorchLearner
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.multiprocessing import spawn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer
import torch.distributed.algorithms.model_averaging.averagers as averagers
import os
import random
import numpy as np
from multiprocessing import current_process

T = TypeVar('T', bound=TorchLearner)
C = TypeVar('C', bound=FullCourse)

class TorchTrainer(Trainer):

    def __init__(self, course:C=None, learner:T=None, conf=None, grader=None, logger=None) -> None:
        super(TorchTrainer, self).__init__(course, learner, conf, grader, logger)

        if not self.conf.distributed or self.conf.distributed.type == 'none':
            self._dist = False
        else: # distributed learning set
            self._dist = True

        self.seed = conf.seed
        self.unit = conf.unit
        self.save_interval = conf.save_interval
        self.compile = conf.accel
        self.init_step = conf.init_step
        self.max_step = conf.max_step

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
        if not force and (self.unit != unit or timestep % self.save_interval()):
            return
        self._save_model(timestep)
        self._save_train_state(timestep)
        self.log.info(f"Checkpoint at {timestep} {unit} saved.")
        
    def dist(self, local_rank:int, fn:str, *args) -> None:
        if self.conf.distributed.debug:
            os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.conf.distributed.type == "torchrun":
            local_size = self.conf.distributed.local_size = int(os.environ['LOCAL_WORLD_SIZE'])
            self.conf.distributed.local_rank = int(os.environ['LOCAL_RANK'])
            self.conf.distributed.global_rank = int(os.environ['RANK'])
            self.conf.distributed.world_size = int(os.environ['WORLD_SIZE'])
            init_process_group(backend=self.conf.distributed.backend)
            self.learner.model = DDP(self.learner.model).compile() if self.compile else DDP(self.learner.model)
            self.learner.opt = self.get_dist_opt(self.conf.distributed.optimizer, self.learner.model.get_params())
            getattr(self, fn)(self.conf.distributed.local_rank)
        elif self.conf.distributed.type == "DDP":
            print(current_process().name)
            os.environ["MASTER_ADDR"] = self.conf.distributed.master_address
            os.environ["MASTER_PORT"] = self.conf.distributed.master_port
            print(local_rank)
            rank = self.conf.distributed.global_rank + local_rank
            init_process_group(backend=self.conf.distributed.backend, world_size=self.conf.distributed.world_size, rank=rank)
            self.learner.model = DDP(self.learner.model).compile() if self.compile else DDP(self.learner.model)
            self.learner.opt = self.get_dist_opt(self.conf.distributed.optimizer, self.learner.model.get_params())
            return getattr(self, fn)(local_rank, *args)
        else:
            raise ValueError("The type of distributed config must be one of the following literals: ['torchrun', 'DDP', 'none']")

    def distributed (fn:Callable):
        def wrapper(*args):
            self = args[0]
            self.loader = self.get_loader(fn.__name__)
            with logging_redirect_tqdm():
                if self._dist:
                    if self.conf.distributed.type == "DDP":
                        if current_process().name == "MainProcess":
                            spawn(self.dist, (fn.__name__),nprocs=self.conf.distributed.local_size, join=True)
                    return self.dist(os.environ['LOCAL_RANK'], fn.__name__, *args)
                else:
                    return fn(*args)
        return wrapper
    
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
        print(current_process().name)
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
                    self._save_checkpoint('step' + current_step)
                current_step += 1
            self.learner.epoch()
            self.log.info(f"Epoch {epoch} finished")
            #self.log.log_state()

        self._save_checkpoint('final')

        if self.distributed:
            destroy_process_group()

    @distributed
    def eval(self, local_rank:int, final:bool=False, silent:bool=False):
        # Evaluation
        if not self.course:
            raise AttributeError("No evaluation dataset.")
            
        whole_batch = None
        for batch in tqdm(self.course, desc='Validation', unit='step', leave=False, disable=silent):
            output = self.learner.infer(batch, local_rank)
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
    def predict(self, local_rank:int, final:bool=False, silent:bool=False):
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
    
    def accelerate(self):
        self.learner.set_model(torch.compile(self.learner.model))

    def get_dist_opt(self, method:Literal['zero', 'postlocal'], params=None, **kwargs):
        if method == 'zero':
            if not params:
                raise ValueError("model parameters require for zero redundancy optimizer.")
            return ZeroRedundancyOptimizer(params, optimizer_class=getattr(torch.optim, type(self.learner.opt).__name__), parameters_as_bucket_view=False, overlap_with_ddp=False, **kwargs)
        if method == 'postlocal':
            return PostLocalSGDOptimizer(self.learner.opt, averagers.PeriodicModelAverager(**kwargs))
        
    def get_loader(self, dataset:Literal['fit', 'eval', 'predict']) -> DataLoader:
        dataset = self.course.get_dataset(dataset)
        return DataLoader(dataset, batch_size=self.conf.batch_size, num_workers=self.conf.num_workers) if not self.distributed else \
        DataLoader(dataset, batch_size=self.conf.batch_size,  num_workers=self.conf.num_workers, sampler=DistributedSampler(dataset, self.conf.distributed.world_size, self.conf.distributed.global_rank))