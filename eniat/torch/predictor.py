from typing import TypeVar, Union, Sequence, Callable, Literal
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import always_wrap_policy, size_based_auto_wrap_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy, enable_wrap, wrap
from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer
import torch.distributed.algorithms.model_averaging.averagers as averagers
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from ..data import get_course_instance
from .learner import load_learner
import os
import warnings
import functools

class Hooker:
    def __init__(
            self,
            labels:Sequence[str]=['before_epoch', 'before_step', 'after_step', 'after_epoch'],
            hooks:dict=None
            ) -> None:
        
        self.hooks = { label : [] for label in labels }
        if hooks is not None:
            self.update(hooks)
    
    def hook(self, label:str, func:Callable) -> None:
        self.hooks[label].append(func)

    def pull(self, label:str, *args, **kwargs) -> None:
        for hook in self.hooks:
            hook(*args, **kwargs)

    def update(self, hooks:dict) -> None:
        for hook, func in hooks.items():
            if hook in self.hooks and isinstance(func, list):
                self.hooks[hook] += func

    def add_label(self, label:str) -> None:
        self.hooks[label] = []

def to_tensor(batch, dtype:str=None, device=None):
    if isinstance(batch, list):
        for data in batch:
            data = torch.as_tensor(data, dtype=getattr(torch, dtype) if dtype else torch.float32, device=device)
    else:
        batch = torch.as_tensor(batch, dtype=getattr(torch, dtype) if dtype else torch.float32, device=device)
    return batch

def distributed (fn:Callable) -> Callable:
    def wrapper(self, device=None, global_rank=None, silent=False, position=0, final=True, **kwargs):

        if not dist.is_initialized():

            if self.env.type == "single": # default
                device = device if (device is not None) else (self.env.dev_id if (isinstance(self.env.dev_id, int)) else self.env.dev_id[0])
                return fn(self, device, device, silent, position, final, **(kwargs or {}))
            elif self.env.type=="DP": # DP (Data Parallel)
                return fn(self, 'cuda', 0, silent, position, final, **(kwargs or {}))

            if self.env.debug:
                os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

            if dist.is_torchelastic_launched():
                #torchrun
                self.hc = HydraConfig.get()
                self._torchrun_init(self)
                return fn(int(os.environ['LOCAL_RANK']), int(os.environ['RANK']), silent, position, final, **(kwargs or {}))
            else:
                #DDP or FSDP
                spawn(self._ddp_init, (fn.__name__, HydraConfig.get(), silent, position, final, *[kwargs[key] for key in kwargs]), nprocs=self.env.local_size, join=True)
        else:
            if dist.get_rank() == 0:
                return fn(self, device, global_rank, silent, position, final, **(kwargs or {}))
            else:
                warnings.simplefilter("ignore")
                with self.log.silent():
                    return fn(self, device, global_rank, True, position, final, **(kwargs or {}))
                
    return wrapper

class TorchPredictor():

    def __init__(self, env_configuration:DictConfig, hooks:dict=None) -> None:
        self.env = env_configuration
        self.hooker = Hooker()

    def get_loader(self, data_label:str, dataset=None) -> DataLoader:
        if dataset is None:
            dataset = self.course.get_dataset(data_label)
        if self.env.type != 'single' and self.env.type != 'DP':
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.loader.shuffle, num_workers=self.loader.num_workers, pin_memory=self.loader.pin_memory) if not dist.is_initialized() else \
            DataLoader(dataset, batch_size=self.batch_size, shuffle=self.loader.shuffle, num_workers=self.loader.num_workers, pin_memory=self.loader.pin_memory, sampler=DistributedSampler(dataset, num_replicas=self.env.world_size, rank=self.env.global_rank))
        else:
            return DataLoader( dataset, shuffle=self.loader.shuffle, num_workers=self.loader.num_workers, pin_memory=self.loader.pin_memory)

    def _ddp_init(self, local_rank:int, fname:str, _hc=None, silent:bool=False, position:int=0, final:bool=False, *args) -> None:
        configure_log(_hc.job_logging, _hc.verbose)
        self.hc = _hc
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = self.env.master_address
        os.environ["MASTER_PORT"] = self.env.master_port
        rank = self.env.global_rank + local_rank
        dist.init_process_group(backend=self.env.backend, init_method=self.env.init_method, world_size=self.env.world_size, rank=rank)
        if not local_rank:
            self.log.info("configured DDP environment...")
        return getattr(self, fname)(local_rank, rank, silent, position, final, *args)

    def _torchrun_init(self):
        self.log.info("setting torchrun environment...")
        self.env.local_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self.env.local_rank = int(os.environ['LOCAL_RANK'])
        self.env.global_rank = int(os.environ['RANK'])
        self.env.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend=self.env.backend)
    
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
            data_cfg=None,
            log=None,
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
            self.learner, state = load_learner(learner_cfg, log, self.env.type, resume_model, resume_opt, resume_dir, resume_step)
            model = self.learner.model

            if self.env.type != 'single' and self.env.type != 'DP':
                if self.env.type == 'FSDP':
                    if 'fsdp_policy' in self.env:
                        if self.env.fsdp_policy == 'always':
                            policy = functools.partial(getattr(torch.distributed.fsdp.wrap , "always_wrap_policy"), **self.env.fsdp_policy_options)
                        elif self.env.fsdp_policy == 'size':
                            policy = functools.partial(getattr(torch.distributed.fsdp.wrap , "size_based_auto_wrap_policy"), **self.env.fsdp_policy_options)
                        elif self.env.fsdp_policy == 'lambda':
                            policy = functools.partial(getattr(torch.distributed.fsdp.wrap , "lambda_auto_wrap_policy"), **self.env.fsdp_policy_options)
                        elif self.env.fsdp_policy == "transformer":
                            policy = functools.partial(getattr(torch.distributed.fsdp.wrap , "transformer_auto_wrap_policy"), **self.env.fsdp_policy_options) 
                            policy = getattr()
                        model = torch.compile(FSDP(model.to(device)), fsdp_auto_wrap_policy=policy) if compile else FSDP(model.to(device), fsdp_auto_wrap_policy=policy)
                    else:
                        model = torch.compile(FSDP(model.to(device))) if compile else FSDP(model)
                else: #DDP
                    model = torch.compile(DDP(model.to(device))) if compile else DDP(model)
                optim = self.get_dist_opt(dist_opt, self.learner.opt, model.parameters())
                self.learner.opt = optim
            else:
                if self.env.type == "DP": # DP (Data Parallel)
                    model = torch.compile(DP(model)).to(device) if compile else DP(model).to(device)
                else: # Default
                    model = torch.compile(model).to(device) if compile else model.to(device)

            self.learner.model = model

        return state or None

    @distributed
    def predict(self, device:Union[int, str], global_rank:int=None, silent:bool=False, position=0, final:bool=True, data_label:str='predict', skip_prepare:bool=False):
        if not skip_prepare:
            self.prepare(device, data_label, self.accel, self.learner_conf, self.data_conf, self.log, self.env.optimizer if 'optimizer' in self.env else None)

        ret = []
        gt = []
        y = None

        with logging_redirect_tqdm():
            for batch in (step_bar:=tqdm(self.loader, desc='Inference', unit='step', position=position, leave=False, disable=silent)):

                batch = to_tensor(batch, self.dtype, device)
                if isinstance(batch, list):
                    y = batch[1].cpu()
                    batch = batch[0]
                self.learner.to(device)
                self.learner.eval()
                with torch.no_grad():
                    if self.precision is None:
                        pred = self.learner.predict(batch, device, self.log)
                    else:
                        with torch.autocast(torch.device(device), dtype=getattr(torch, self.precision)):
                            pred = self.learner.predict(batch, device, self.log)

                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        gathered = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
                        dist.gather(pred, gathered)
                        ret.append(torch.cat(gathered).cpu())
                        dist.barrier()
                        if y is not None:
                            gathered = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
                            dist.gather(y, gathered)
                            gt.append(torch.cat(gathered).cpu())
                            dist.barrier()
                    else:
                        dist.gather(pred)
                        dist.barrier()
                        if y is not None:
                            dist.gather(y)
                            dist.barrier()

        self.hooker.pull('after_epoch')

        if not dist.is_initialized() or dist.get_rank() == 0:
            ret = torch.cat(ret).squeeze().numpy()
            if len(gt):
                gt = torch.cat(gt).squeeze().numpy()
            else:
                gt = None
        else:
            ret = gt = None

        if dist.is_initialized() and final:
            dist.destroy_process_group()

        return (ret, gt)
