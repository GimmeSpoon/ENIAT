from typing import TypeVar, Generic, Union, Any
from abc import abstractmethod
from ..core import Learner
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from ..utils import instantiate, Logger, load_class
from omegaconf import DictConfig
from pathlib import Path

T_co = TypeVar("T_co", covariant=True)
O = TypeVar("O", bound=Optimizer)
L = TypeVar("L", bound=Logger)


def load_learner(learner_conf: DictConfig):
    return load_class(learner_conf.path, learner_conf.cls)(learner_conf)


def instantiate_learner(
    conf: DictConfig,
    log: L = None,
    _model: Any = None,
    loss_fn: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    as_tuple: bool = False,
):
    # Model Load
    if _model is None:
        model = instantiate(conf.model)
        if log is not None:
            log.info("Model loaded...")
    else:
        model = _model

    # loss
    if loss_fn is None:
        loss = instantiate(conf.loss)
    else:
        loss = loss_fn
    if loss:
        if log is not None:
            log.info("Loss function loaded...")
    else:
        if log is not None:
            log.warning("Loss function is not configured.")

    # optimizer
    if optimizer is None:
        optim = instantiate(conf.optimizer, params=model.parameters())
    else:
        optim = optimizer
    if optim:
        if log is not None:
            log.info("Optimizer loaded...")
    else:
        if log is not None:
            log.warning("Optimizer is not configured.")

    # scheduler
    if scheduler is None:
        if conf.scheduler.cls:
            schlr = instantiate(conf.scheduler, optimizer=optim)
        else:
            schlr = None
    else:
        schlr = scheduler

    if schlr:
        if log is not None:
            log.info("Scheduler loaded...")
    else:
        if log is not None:
            log.warning("Scheduler is not configured.")

    # instantiate learner
    learner = instantiate(conf, model, loss, optim, schlr)
    if log is not None:
        log.info("Learner instance created.")

    return (model, loss, optim, schlr) if as_tuple else learner


class TorchLearner(Learner, Generic[T_co]):
    def __init__(
        self,
        conf: DictConfig = None,
        model: Module = None,
        criterion=None,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super(TorchLearner).__init__()
        self.conf = conf
        self.model = model
        self.loss_fn = criterion
        self.opt = optimizer
        self.sch = scheduler

    @abstractmethod
    def fit(self, batch: Tensor, device: int, logger):
        pass

    @abstractmethod
    def predict(self, batch: Tensor, device: int, logger):
        pass

    def prepare(self, log: L = None) -> None:
        self.model, self.loss_fn, self.opt, self.sch = instantiate_learner(
            self.conf, log, as_tuple=True
        )

    def resume_model(self, state=None, path: Union[str, Path] = None) -> None:
        if state:
            self.model.load_state_dict(state)
        else:
            with open(path, "rb") as f:
                self.model.load_state_dict(torch.load(f))

    def resume_optimizer(self, state=None, path: Union[str, Path] = None) -> None:
        if state:
            self.opt.load_state_dict(state)
        else:
            with open(path, "rb") as f:
                self.opt.load_state_dict(torch.load(f)["optimizer"])

    def resume_scheduler(self, state=None, path: Union[str, Path] = None) -> None:
        if state:
            self.sch.load_state_dict(state)
        else:
            with open(path, "rb") as f:
                self.sch.load_state_dict(torch.load(f)["scheduler"])

    def resume(self) -> bool:
        if self.conf.resume_path:
            if isinstance(self.conf.resume_path, str):
                resume_path = Path(self.conf.resume_path)
            if resume_path.is_file():
                self.resume_model(path=resume_path)
            else:
                raise FileNotFoundError(
                    f"Cannot open the checkpoint: {self.conf.resume_path}"
                )
        elif self.conf.resume_dir and self.conf.resume_step:
            if isinstance(self.conf.resume_dir, str):
                resume_dir = Path(self.conf.resume_dir)
            if resume_dir.is_dir() and isinstance(self.conf.resume_step, int):
                self.resume_model(
                    path=f"{resume_dir.joinpath(f'model_{self.conf.resume_step}.cpt')}"
                )
        else:
            return False
        return True

    @property
    def get_model(self):
        return self.model

    @property
    def get_optimizer(self):
        return self.opt

    @property
    def get_state(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.op.state_dict() if self.opt else None,
        }

    def to(self, device: Union[int, str]):
        self.model.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class SupervisedLearner(TorchLearner):
    def __init__(
        self, model: Module, criterion=None, optimizer=None, scheduler=None
    ) -> None:
        super().__init__(model, criterion, optimizer, scheduler)

    def fit(self, batch: Tensor, device: int, logger):
        x, y = batch
        x, y = x.to(device), y.to(device)
        model = self.model.to(device)
        return self.loss_fn(model(x).squeeze(), y.squeeze())

    def predict(self, batch: Tensor, device: int, logger):
        batch = batch.to(device)
        self.model.to(device)
        return self.model(batch)
