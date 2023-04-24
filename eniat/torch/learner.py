from typing import TypeVar, Generic
from abc import abstractmethod
from ..base import Learner
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from hydra.utils import instantiate
from ..utils.conf import conf_instantiate, _dynamic_import
from importlib import import_module

T_co = TypeVar('T_co', covariant=True)
O = TypeVar('O', bound=Optimizer)

def load_learner (conf, log):
    # Model Load
    model = conf_instantiate(conf.learner.model)
    log.info("Model loaded...")

    if not conf.learner.resume:
        log.warning("'resume' is set to False. The model will be initialized without loading a checkpoint.")
    # loss
    loss = instantiate(conf.learner.loss) if conf.learner.loss and conf.learner.loss._target_ else None
    if loss:
        log.info("Loss function loaded...")
    else:
        log.warning("Loss function is not defined. Are you sure you wanted this?")
    # optimizer
    optim = instantiate(conf.learner.optimizer, params=model.parameters()) if conf.learner.optimizer and conf.learner.optimizer._target_ else None
    if optim:
        log.info("Optimizer loaded...")
    else:
        log.warning("Optimizer is not defined. Are you sure you wanted this?")
    # scheduler
    schlr = None# instantiate(conf.learner.scheduler, lr_lambda=lambda x: x**conf.learner.scheduler.lr_lambda, optimizer=optim) if conf.learner.scheduler and conf.learner.scheduler._target_ else None
    if schlr:
        log.info("Scheduler loaded...")
    else:
        log.warning("Scheduler is not defined. Edit the configuration if this is not what you wanted.")
    
    # instantiate learner
    if 'path' in conf.learner and conf.learner.path:
        _mod, _bn = _dynamic_import(conf.learner.path)
        learner = getattr(_mod, conf.learner.cls)(model, loss, optim, schlr, conf.learner.resume, conf.learner.resume_path)
    else:
        learner = getattr(import_module('.pytorch.learner', 'eniat'), conf.learner.cls)(model, loss, optim, schlr, conf.learner.resume, conf.learner.resume_path)
    if learner:
        log.info("Learner instance created.")

class TorchLearner(Learner, Generic[T_co]):
    def __init__(self, model:Module, criterion=None, optimizer=None, scheduler=None, resume:bool=False, resume_path:str=None) -> None:
        super(TorchLearner).__init__()
        self.model = model
        self.loss_fn = criterion
        self.opt = optimizer
        self.sch = scheduler

        if resume:
            self.load_model(path=resume_path)

    @abstractmethod
    def fit(self, batch:Tensor, device:int, logger):
        pass
    
    @abstractmethod
    def predict(self, batch:Tensor, device:int, logger):
        pass

    def load_model(self, state=None, path:str=None) -> None:
        if state:
            self.model.load_state_dict(state)
        else:
            with open(path, 'rb') as f:
                self.model.load_state_dict(torch.load(f))

    def load_optimizer(self, state=None, path:str=None) -> None:
        if state:
            self.opt.load_state_dict(state)
        else:
            with open(path, 'rb') as f:
                self.opt.load_state_dict(torch.load(f))

    @property
    def get_model(self):
        return self.model

    @property
    def get_optimizer(self):
        return self.opt
    
    @property
    def get_state(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.op.state_dict() if self.opt else None
        }

class SupremeLearner (TorchLearner):
    def __init__(self, model: Module, criterion=None, optimizer=None, scheduler=None, resume: bool = False, resume_path: str = None) -> None:
        super().__init__(model, criterion, optimizer, scheduler, resume, resume_path)

    def fit(self, batch: Tensor, device: int, logger):
        x, y = batch
        x, y = x.to(device), y.to(device)
        model = self.model.to(device)
        pred = model(x)
        return self.loss_fn(pred, y)
    
    def predict(self, batch: Tensor, device: int, logger):
        batch = batch.to(device)
        model = self.model.to(device)
        pred = model(x)
        return pred