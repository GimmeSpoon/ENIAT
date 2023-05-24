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
    def wrapper(self, device=None, global_rank=None, silent=False, position=0, final=True, **kwargs):
        if not dist.is_initialized():

            if self.conf.env.type == "single":
                device = device if (device is not None) else (self.conf.env.dev_id if (isinstance(self.conf.env.dev_id, int)) else self.conf.env.dev_id[0])
                return fn(self, device, device, silent, position, final, **(kwargs or {}))
            elif self.conf.env.type=="DP":
                return fn(self, 'cuda', 0, silent, position, final, **(kwargs or {}))

            if self.conf.env.debug:
                os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

            if dist.is_torchelastic_launched():
                #torchrun
                self.hc = HydraConfig.get()
                self._torchrun_init(self)
                return fn(int(os.environ['LOCAL_RANK']), int(os.environ['RANK']), silent, position, final, **(kwargs or {}))
            else:
                #DDP
                spawn(self._ddp_init, (fn.__name__, HydraConfig.get(), silent, position, final, *[kwargs[key] for key in kwargs]), nprocs=self.conf.env.local_size, join=True)
        else:
            if dist.get_rank() == 0:
                return fn(self, device, global_rank, silent, position, final, **(kwargs or {}))
            else:
                warnings.simplefilter("ignore")
                with self.log.silent():
                    return fn(self, device, global_rank, True, position, final, **(kwargs or {}))
                
    return wrapper

class TorchPredictor():

    def get_loader(self, data_label:str, dataset=None) -> DataLoader:
        if dataset is None:
            dataset = self.course.get_dataset(data_label)
        if self.conf.env.type != 'single' and self.conf.env.type != 'DP':
            return DataLoader(dataset, batch_size=self.conf.batch_size, num_workers=self.conf.num_workers) if not dist.is_initialized() else \
            DataLoader(dataset, batch_size=self.conf.batch_size,  num_workers=self.conf.num_workers, sampler=DistributedSampler(dataset, self.conf.env.world_size, self.conf.env.global_rank))
        else:
            return DataLoader( dataset, num_workers=self.conf.num_workers)

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
    
    def prepare (
            self,
            device:int,
            data_label:str=None,
            compile:bool=False,
            learner_cfg=None,
            data_cfg=None, log=None,
            resume_model:bool=False,
            resume_opt:bool=False,
            resume_dir:str=None,
            resume_step:int=None,
            dist_opt:str=None
            ):
        # data
        if data_cfg is not None and data_label is not None:
            self.course = get_course_instance(data_cfg, log)
            self.loader = self.get_loader(data_label)

        # learner
        if learner_cfg is not None:
            self.learner, state = load_learner(learner_cfg, log, resume_model, resume_opt, resume_dir, resume_step)
        model = self.learner.model

        if self.conf.env.type != 'single' and self.conf.env.type != 'DP':
            model = torch.compile(DDP(model.to(device))) if compile else DDP(model)
            optim = self.get_dist_opt(dist_opt, self.learner.opt, model.parameters())
            self.learner.opt = optim
        else:
            if self.conf.env.type == "DP":
                model = torch.compile(DP(model)).to(device) if compile else DP(model).to(device)
            else:
                model = torch.compile(model).to(device) if compile else model.to(device)

        self.learner.model = model

        return state or None

    @distributed
    def predict(self, device:Union[int, str], global_rank:int=None, silent:bool=False, position=0, final:bool=True, data_label:str='predict', keep_learner:bool=False):
        
        self.prepare(device, data_label, self.conf.accel, self.learner_conf if keep_learner else self.learner_conf, self.data_conf, self.log, self.conf.env.optimizer if 'optimizer' in self.conf.env else None)

        ret = []
        gt = []
        with logging_redirect_tqdm():
            for batch in (step_bar:=tqdm(self.loader, desc='Inference', unit='step', position=position, leave=False, disable=silent)):
                if isinstance(batch, list):
                    gt.append(batch[1].cpu())
                    batch = to_tensor(batch[0])
                batch = self.to_tensor(batch)
                self.learner.to(device)
                self.learner.eval()
                with torch.no_grad():
                    if self.conf.precision is None:
                        pred = self.learner.predict(batch).detach().cpu()
                    else:
                        with torch.autocast(torch.device(device), dtype=getattr(torch, self.conf.precision)):
                            pred = self.learner.predict(batch)
                        pred = pred.detach().cpu()

                if dist.is_initialized() and dist.get_rank() == 0:
                    gathered = torch.zeros((dist.get_world_size() * pred[0], *pred.shape[1:]))
                    dist.all_gather_into_tensor(gathered, pred)
                    ret.append(gathered)
                else:
                    ret.append(pred)

        if dist.is_initialized() and dist.get_rank():
            dist.destroy_process_group()
            return
        
        ret = torch.cat(ret).numpy()
        if gt:
            gt = torch.cat(gt).numpy()
        else:
            gt = None

        if dist.is_initialized() and final:
            dist.destroy_process_group()
        return ret if gt is None else (ret, gt)
