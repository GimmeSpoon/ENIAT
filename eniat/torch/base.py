from typing import TypeVar, Union, Sequence, Callable, Literal
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer
import torch.distributed.algorithms.model_averaging.averagers as averagers
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from ..data import get_course_instance
from .learner import load_learner
from abc import abstractmethod
import numpy as np
import os
import warnings

def to_tensor(batch):
    if isinstance(batch, list):
        for data in batch:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
    elif isinstance(batch, np.ndarray):
        batch = torch.from_numpy(data)
    return batch

def distributed (fn:Callable) -> Callable:
    def wrapper(self, device=None, global_rank=None, silent=False, position=0):
        if not dist.is_initialized():

            if self.conf.env.type == "single":
                device = device if (device is not None) else (self.conf.env.dev_id if (isinstance(self.conf.env.dev_id, int)) else self.conf.env.dev_id[0])
                return fn(self, device, device, silent, position)
            elif self.conf.env.type=="DP":
                return fn(self, 'cuda', 0, silent, position)

            if self.conf.env.debug:
                os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

            if dist.is_torchelastic_launched():
                #torchrun
                self.hc = HydraConfig.get()
                self._torchrun_init(self)
                return fn(int(os.environ['LOCAL_RANK']), int(os.environ['RANK']), silent, position)
            else:
                #DDP
                spawn(self._ddp_init, (fn.__name__, HydraConfig.get(), silent, position), nprocs=self.conf.env.local_size, join=True)
        else:
            if dist.get_rank() == 0:
                return fn(self, device, global_rank, silent, position)
            else:
                warnings.simplefilter("ignore")
                with self.log.silent():
                    return fn(self, device, global_rank, True, position)
                
    return wrapper

class TorchPredictor():

    @abstractmethod
    def get_loader(self):
        raise NotImplementedError

    def _ddp_init(self, local_rank:int, fname:str, _hc=None, silent:bool=False, position:int=0) -> None:
        configure_log(_hc.job_logging, _hc.verbose)
        self.hc = _hc
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = self.conf.env.master_address
        os.environ["MASTER_PORT"] = self.conf.env.master_port
        rank = self.conf.env.global_rank + local_rank
        dist.init_process_group(backend=self.conf.env.backend, init_method=self.conf.env.init_method, world_size=self.conf.env.world_size, rank=rank)
        if not local_rank:
            self.log.info("configured DDP environment...")
        return getattr(self, fname)(local_rank, rank, silent, position)

    def _torchrun_init(self):
        self.log.info("setting torchrun environment...")
        self.conf.env.local_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self.conf.env.local_rank = int(os.environ['LOCAL_RANK'])
        self.conf.env.global_rank = int(os.environ['RANK'])
        self.conf.env.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend=self.conf.env.backend)
    
    def get_dist_opt(self, method:Literal['zero', 'postlocal'], opt=None, params=None, **kwargs):
        if method == 'zero':
            if not params:
                raise ValueError("model parameters require for zero redundancy optimizer.")
            return ZeroRedundancyOptimizer(params, optimizer_class=getattr(torch.optim, type(opt).__name__), parameters_as_bucket_view=False, overlap_with_ddp=False, **kwargs)
        if method == 'postlocal':
            return PostLocalSGDOptimizer(opt, averagers.PeriodicModelAverager(**kwargs))
        raise ValueError(f'Distributed optimizer "{method}" is not valid. Configure as one of following choices:[zero, postlocal]')
    
    def prepare (self, device:int, task:Literal['fit', 'eval', 'predict'], compile, learner_cfg, data_cfg, log=None, dist_opt:str=None):
        # data
        self.course = get_course_instance(data_cfg, log)
        self.loader = self.get_loader(task)

        # learner
        self.learner = load_learner(learner_cfg, log)
        model = self.learner.model

        if dist.is_initialized():
            model = torch.compile(DDP(model.to(device))) if compile else DDP(model)
            optim = self.get_dist_opt(dist_opt, self.learner.opt, model.parameters())
            self.learner.opt = optim
        else:
            if self.conf.env.type == "DP":
                model = torch.compile(DP(model)).to(device) if compile else DP(model).to(device)
            else:
                model = torch.compile(model).to(device) if compile else model.to(device)

        self.learner.model = model
    
    @distributed
    def predict(self, device:Union[int, str], global_rank:int=None, silent:bool=False, position=0):
        self.prepare(device, 'predict', self.conf.accel, self.learner_conf, self.data_conf, self.log, self.conf.env.optimizer if 'optimizer' in self.conf.env else None)

        ret = []
        with logging_redirect_tqdm():
            for batch in (step_bar:=tqdm(self.loader, desc='Inference', unit='step', position=position, leave=False, disable=silent)):
                batch = self.to_tensor(batch)
                self.learner.to(device)
                self.learner.eval()
                with torch.no_grad():
                    pred = self.learner.predict(batch).detach().cpu()

                if dist.is_initialized() and dist.get_rank() == 0:
                    gathered = torch.zeros((dist.get_world_size() * pred[0], *pred.shape[1:]))
                    dist.all_gather_into_tensor(gathered, pred)
                    ret.append(gathered)
                else:
                    ret.append(pred)

        if dist.is_initialize() and dist.get_rank():
            dist.destroy_process_group()
            return
        ret = torch.cat(ret).numpy()
        dist.destroy_process_group()
        return ret
