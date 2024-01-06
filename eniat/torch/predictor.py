import contextlib
import functools
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence, TypeVar, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
from omegaconf import DictConfig
from torch.distributed.optim import (
    DistributedOptimizer,
    PostLocalSGDOptimizer,
    ZeroRedundancyOptimizer,
)
from torch.multiprocessing import spawn
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..utils import Logger, advance, bar, end, instantiate

L = TypeVar("L", bound=Logger)

ENV_NON_DIST_TYPES = ("single", "DP")
ENV_DIST_TYPES = ("DDP", "torchrun", "FSDP")


class Hooker:
    def __init__(
        self,
        labels: Sequence[str] = [
            "before_epoch",
            "before_step",
            "after_step",
            "after_epoch",
        ],
        hooks: dict = None,
    ) -> None:

        self.hooks = {label: [] for label in labels}
        if hooks is not None:
            self.update(hooks)

    def hook(self, label: str, func: Callable) -> None:
        self.hooks[label].append(func)

    def pull(self, label: str, *args, **kwargs) -> None:
        for hook in self.hooks:
            hook(*args, **kwargs)

    def update(self, hooks: dict) -> None:
        for hook, func in hooks.items():
            if hook in self.hooks and isinstance(func, list):
                self.hooks[hook] += func

    def add_label(self, label: str) -> None:
        self.hooks[label] = []


def to_tensor(batch, dtype: str = None, device=None):
    if isinstance(batch, list):
        for data in batch:
            data = torch.as_tensor(
                data,
                dtype=getattr(torch, dtype) if dtype else torch.float32,
                device=device,
            )
    else:
        batch = torch.as_tensor(
            batch,
            dtype=getattr(torch, dtype) if dtype else torch.float32,
            device=device,
        )
    return batch


def distributed(fn: Callable) -> Callable:
    def wrapper(
        self,
        device=None,
        global_rank=None,
        silent=False,
        final=True,
        **kwargs,
    ):

        if not dist.is_initialized():

            if self.conf.env.type == "single":  # default
                device = (
                    device
                    if (device is not None)
                    else (
                        self.conf.env.device_id
                        if (isinstance(self.conf.env.device_id, int))
                        else self.conf.env.device_id[0]
                    )
                )
                return fn(self, device, device, silent, final, **(kwargs or {}))
            elif self.conf.env.type == "DP":  # DP (Data Parallel)
                return fn(self, "cuda", 0, silent, final, **(kwargs or {}))

            if self.conf.env.debug:
                os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

            if dist.is_torchelastic_launched():
                # torchrun
                self._torchrun_init(self)
                return fn(
                    int(os.environ["LOCAL_RANK"]),
                    int(os.environ["RANK"]),
                    silent,
                    final,
                    **(kwargs or {}),
                )
            else:
                # DDP or FSDP
                spawn(
                    self._ddp_init,
                    (
                        fn.__name__,
                        silent,
                        final,
                        *[kwargs[key] for key in kwargs],
                    ),
                    nprocs=self.conf.env.dist.local_size,
                    join=True,
                )
        else:
            if dist.get_rank() == 0:
                return fn(self, device, global_rank, silent, final, **(kwargs or {}))
            else:
                with contextlib.redirect_stdout(None):
                    with contextlib.redirect_stderr(None):
                        return fn(
                            self,
                            device,
                            global_rank,
                            True,
                            final,
                            **(kwargs or {}),
                        )

    return wrapper


class TorchPredictor:
    def __init__(self, env_configuration: DictConfig, hooks: dict = None) -> None:
        self.hooker = Hooker()

    def __rank0(self):
        return dist.is_initialized() and dist.get_rank() == 0

    def get_loader(self, data_label: str = None, dataset=None) -> DataLoader:
        if dataset is None:
            if isinstance(data_label, str):
                dataset = self.course.get(data_label)
            else:
                raise ValueError(f"Invalid data label: {data_label}")
        if self.conf.env.type in ENV_DIST_TYPES:
            return (
                DataLoader(
                    dataset,
                    batch_size=self.conf.loader.batch_size,
                    shuffle=self.conf.loader.shuffle,
                    num_workers=self.conf.loader.num_workers,
                    pin_memory=self.conf.loader.pin_memory,
                )
                if not dist.is_initialized()
                else DataLoader(
                    dataset,
                    batch_size=self.conf.loader.batch_size,
                    shuffle=self.conf.loader.shuffle,
                    num_workers=self.conf.loader.num_workers,
                    pin_memory=self.conf.loader.pin_memory,
                    sampler=DistributedSampler(
                        dataset,
                        num_replicas=self.conf.env.dist.world_size,
                        rank=self.conf.env.dist.global_rank,
                    ),
                )
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.conf.loader.batch_size,
                shuffle=self.conf.loader.shuffle,
                num_workers=self.conf.loader.num_workers,
                pin_memory=self.conf.loader.pin_memory,
            )

    def _ddp_init(
        self,
        local_rank: int,
        fname: str,
        silent: bool = False,
        final=True,
        *args,
    ) -> None:
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = self.conf.env.dist.master_address
        os.environ["MASTER_PORT"] = str(self.conf.env.dist.master_port)
        rank = self.conf.env.dist.global_rank + local_rank
        dev_id = 0
        if self.conf.env.device_id is not None:
            if isinstance(self.conf.env.device_id, list):
                dev_id = self.conf.env.device_id[local_rank]
            else:
                dev_id = local_rank + self.conf.env.device_id
        else:
            dev_id = local_rank
        dist.init_process_group(
            backend=self.conf.env.dist.backend,
            init_method=self.conf.env.dist.init_method,
            world_size=self.conf.env.dist.world_size,
            rank=rank,
        )
        if local_rank == 0:
            self.log.info("configured DDP environment...")
        return getattr(self, fname)(dev_id, rank, silent, final, *args)

    def _torchrun_init(self):
        self.log.info("setting torchrun environment...")
        self.conf.env.dist.local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.conf.env.dist.local_rank = int(os.environ["LOCAL_RANK"])
        self.conf.env.dist.global_rank = int(os.environ["RANK"])
        self.conf.env.dist.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=self.conf.env.dist.backend)

    def get_dist_opt(self, method: str, opt=None, params=None, _lr=None, **kwargs):
        if method == "zero":
            if not params:
                raise ValueError(
                    "model parameters require for zero redundancy optimizer."
                )
            return ZeroRedundancyOptimizer(
                params,
                optimizer_class=getattr(torch.optim, type(opt).__name__),
                lr=_lr,
                **kwargs,
            )
        if method == "postlocal":
            return PostLocalSGDOptimizer(opt, averagers.PeriodicModelAverager(**kwargs))
        if method == "default":
            return DistributedOptimizer(opt, params, lr=_lr, **kwargs)
        if method is None or method == "none":
            return opt
        raise ValueError(
            f'Distributed optimizer "{method}" is not valid.'
            "Configure as one of following choices:[zero, postlocal]"
        )

    def rand_all(self, seed) -> None:
        if self.conf.env.type != "single":
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        self.log.debug(f"Random seed set:{seed}")

    def get_rand_state(self) -> dict:
        return {
            "cuda": torch.cuda.get_rng_state_all()
            if self.conf.env.type != "single"
            else torch.cuda.get_rng_state(),
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    def set_rand_state(self, state: dict) -> None:
        if self.conf.env.type != "single":
            torch.cuda.set_rng_state_all(state["cuda"])
            for i, rng_state in enumerate(state["cuda"]):
                torch.cuda.set_rng_state(rng_state)
        else:
            torch.cuda.set_rng_state(state["cuda"])
        torch.set_rng_state(state["torch"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.set_state(state["numpy"])
        random.setstate(state["random"])
        self.log.debug("Random state set.")

    def load_state(self, path: str, force:bool=False) -> None:
        resumed = torch.load(path)
        if resumed["scheduler"]:
            self.resume_scheduler(resumed["scheduler"])
        self.resume_optimizer(resumed["optimizer"])
        self.set_rand_state(resumed["rng_state"])
        if force or not self.conf.scheme.unit:
            self.conf.scheme.unit = resumed["unit"]
        if force or not self.conf.scheme.init_iter:
            self.conf.scheme.init_iter = resumed["timestep"]
        if force or not self.conf.loader.batch_size:
            self.conf.scheme.batch_size = resumed["batch_size"]
        if force or not self.conf.loader.total_iters:
            self.conf.scheme.total_iters = resumed["total_iters"]
        self.conf.env = resumed["env"]
        self.log.debug("Random state loaded.")

    def resume_optimizer(self, state_dict=None, path:Union[str, Path]=None) -> None:
        if state_dict:
            self.opt.load_state_dict(state_dict)
        else:
            with open(path, "rb") as f:
                self.opt.load_state_dict(torch.load(f)["optimizer"])

    def resume_scheduler(self, state_dict=None, path: Union[str, Path] = None) -> None:
        if state_dict:
            self.sch.load_state_dict(state_dict)
        else:
            with open(path, "rb") as f:
                self.sch.load_state_dict(torch.load(f)["scheduler"])

    def prepare(
        self,
        device: Union[int, str],
        log=None,
        load_data: bool = True,
        load_learner: bool = True,
    ):

        # logger
        if dist.is_initialized():
            _rank = dist.get_rank()
            log.reload(log.name + f".rank{_rank}")
            if _rank:
                log.be_inactive()
            else:
                log.load_rich_console()
        log.debug("Logger ready.")

        # data
        if load_data:
            self.course.load()
            self.loader = self.get_loader(self.conf.data_label)
        log.debug("Data ready.")

        # learner
        if load_learner:
            self.learner.prepare()
            model = self.learner.model.to(device)
            log.debug("Model loaded.")

            if self.conf.env.type in ENV_DIST_TYPES:
                if self.conf.env.type == "FSDP":

                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp.fully_sharded_data_parallel import (
                        BackwardPrefetch,
                        CPUOffload,
                    )
                    from torch.distributed.fsdp.wrap import (
                        always_wrap_policy,
                        enable_wrap,
                        lambda_auto_wrap_policy,
                        size_based_auto_wrap_policy,
                        transformer_auto_wrap_policy,
                        wrap,
                    )

                    if self.conf.env.dist.fsdp_policy:
                        if self.conf.env.dist.fsdp_policy == "always":
                            policy = functools.partial(
                                always_wrap_policy,
                                **self.conf.env.dist.fsdp_policy_options,
                            )
                        elif self.conf.env.dist.fsdp_policy == "size":
                            policy = functools.partial(
                                size_based_auto_wrap_policy,
                                **self.conf.env.dist.fsdp_policy_options,
                            )
                        elif self.conf.env.dist.fsdp_policy == "lambda":
                            policy = functools.partial(
                                lambda_auto_wrap_policy,
                                **self.conf.env.dist.fsdp_policy_options,
                            )
                        elif self.conf.env.dist.fsdp_policy == "transformer":
                            policy = functools.partial(
                                transformer_auto_wrap_policy,
                                **self.conf.env.dist.fsdp_policy_options,
                            )
                        self.learner.model = (
                            torch.compile(FSDP(model, auto_wrap_policy=policy))
                            if self.conf.scheme.compile
                            else FSDP(
                                model,
                                auto_wrap_policy=policy,
                            )
                        )
                    else:
                        self.learner.model = (
                            torch.compile(FSDP(model))
                            if self.conf.scheme.compile
                            else FSDP(model)
                        )
                else:  # DDP
                    self.learner.model = (
                        torch.compile(DDP(model))
                        if self.conf.scheme.compile
                        else DDP(model)
                    )
                log.debug("Distributed wrapping complete.")

            else:
                if self.conf.env.type == "DP":  # DP (Data Parallel)
                    self.learner.model = (
                        torch.compile(DP(model))
                        if self.conf.scheme.compile
                        else DP(model)
                    )
                    log.debug("Data Parallel wrapping complete.")
                else:  # Single
                    self.learner.model = (
                        torch.compile(model) if self.conf.scheme.compile else model
                    )

            # Components
            if "loss" in self.conf:
                self.loss = instantiate(self.conf.loss)
                log.debug("Loss function loaded.")
            else:
                self.loss = None

            if "optimizer" in self.conf:
                self.opt = instantiate(self.conf.optimizer, self.learner.model.parameters())
                log.debug("Optimizer loaded.")
            else:
                self.opt = None

            if "scheduler" in self.conf and self.conf.scheduler.cls:
                self.sch = instantiate(self.conf.scheduler, self.opt, last_epoch = self.conf.scheme.init_iter - 1)
                log.debug("Scheduler loaded.")
            else:
                self.sch = None

            # Resuming
            if not self.learner.resume():
                log.warning("Model checkpoint not resumed.")

            _res = True
            if "resume_path" in self.conf and self.conf.resume_path is not None:
                if isinstance(self.conf.resume_path, str):
                    resume_path = Path(self.conf.resume_path)
                if resume_path.is_file():
                    self.load_state(path=resume_path)
                    if self.sch:
                        self.sch.last_epoch = self.conf.scheme.init_iter - 1
                else:
                    raise FileNotFoundError(
                        f"No checkpoint(optimizer) exists: {resume_path.absolute()}"
                    )
                log.info(f"Training states resumed from {self.conf.resume_path}.")
            elif "resume_dir" in self.conf and self.conf.resume_dir:
                if isinstance(self.conf.resume_dir, str):
                    resume_dir = Path(self.conf.resume_dir)
                if resume_dir.is_dir():
                    self.load_state(
                        path=resume_dir.joinpath(f"state_{self.conf.resume_step}.cpt")
                    )
                    if self.sch:
                        self.sch.last_epoch = self.conf.scheme.init_iter - 1
                log.info(
                    f"Training states resumed from {resume_dir} at timestep {self.conf.resume_step}."
                )
            else:
                log.warning("Training states not resumed.")
                if self.conf.scheme.seed:
                    self.rand_all(int(self.conf.scheme.seed))
                else:
                    self.log.warning("Random seed is not set.")
                    self.seed = random.random()
                    self.rand_all(self.seed)
                _res = False

            if self.conf.env.type in ENV_DIST_TYPES:
                self.opt = self.get_dist_opt(
                    self.conf.env.dist.optimizer,
                    self.opt,
                    self.learner.model.parameters(),
                    self.conf.optimizer.options.lr,
                )

            return _res

        return False

    @distributed
    def predict(
        self,
        device: Union[int, str],
        global_rank: int = None,
        silent: bool = False,
        final: bool = True,
        msg: str = "Inferencing...",
    ):
        if final:
            self.prepare(device, self.log)

        ret = []
        gt = []
        y = None

        with bar("eval", total_steps=len(self.loader), msg=msg) if (
            not dist.is_initialized() or dist.get_rank() == 0
        ) else nullcontext():

            for _, batch in enumerate(self.loader):

                batch = to_tensor(batch, self.conf.scheme.dtype, device)
                if isinstance(batch, list):
                    y = batch[1].to(device)
                    batch = batch[0]
                self.learner.to(device)
                self.learner.eval()
                with torch.no_grad():
                    if self.conf.scheme.precision is None:
                        pred = self.learner.predict(batch, device, self.log)
                    else:
                        with torch.autocast(
                            torch.device(device),
                            dtype=getattr(torch, self.precision),
                        ):
                            pred = self.learner.predict(batch, device, self.log)

                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        gathered = [
                            torch.zeros_like(pred) for _ in range(dist.get_world_size())
                        ]
                        dist.gather(pred, gathered)
                        ret.append(torch.cat(gathered).cpu())
                        dist.barrier()
                        if y is not None:
                            gathered = [
                                torch.zeros_like(y)
                                for _ in range(dist.get_world_size())
                            ]
                            dist.gather(y, gathered)
                            gt.append(torch.cat(gathered).cpu())
                            dist.barrier()
                    else:
                        dist.gather(pred)
                        dist.barrier()
                        if y is not None:
                            dist.gather(y)
                            dist.barrier()

                if not dist.is_initialized() or dist.get_rank() == 0:
                    advance("eval")

        if not dist.is_initialized() or dist.get_rank() == 0:
            end("eval", not final)
            ret = torch.cat(ret).squeeze()
            if len(gt):
                gt = torch.cat(gt).squeeze()
            else:
                gt = None
        else:
            ret = gt = None

        if dist.is_initialized() and final:
            dist.destroy_process_group()

        if self.conf.scheme.save_inference:
            _dir = Path(self.conf.output_dir)
            torch.save(ret, _dir.joinpath(f"{self.conf.data_label}_inference.pkl"))
            self.log.info("Saved inference result.")

        return (ret, gt)
