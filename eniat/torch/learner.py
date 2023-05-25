from typing import TypeVar, Generic, Union
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
import os

T_co = TypeVar('T_co', covariant=True)
O = TypeVar('O', bound=Optimizer)

def load_learner (conf, log, resume_model:bool=False, resume_opt:bool=False, resume_dir:str = None, resume_step:int=None):
    # Model Load
    model = conf_instantiate(conf.model)
    log.info("Model loaded...")

    if not resume_model:
        log.warning("'resume_model' is set to False. The model will be initialized without loading a checkpoint.")
    else:
        model.load_state_dict(torch.load(model_resume_path:=(os.path.join(resume_dir, f'model_{resume_step}.cpt'))))
        log.info(f"Weights are resumed from {model_resume_path}.")
    # loss
    loss = instantiate(conf.loss) if conf.loss and conf.loss._target_ else None
    if loss:
        log.info("Loss function loaded...")
    else:
        log.warning("Loss function is not defined. Are you sure you wanted this?")
    # optimizer
    optim = instantiate(conf.optimizer, params=model.parameters()) if conf.optimizer and conf.optimizer._target_ else None
    if optim:
        log.info("Optimizer loaded...")
    else:
        log.warning("Optimizer is not defined. Are you sure you wanted this?")
    if not resume_opt:
        log.warning("'resume_opt' is set to False. Training will be initiated without checkpoints.")
    else:
        optim.load_state_dict(ret_state:=(torch.load(opt_resume_path:=(os.path.join(resume_dir, f'state_{resume_step}.cpt')))))
        log.info(f"Training states are resume from {opt_resume_path}.")
    # scheduler
    schlr = None# instantiate(conf.learner.scheduler, lr_lambda=lambda x: x**conf.learner.scheduler.lr_lambda, optimizer=optim) if conf.learner.scheduler and conf.learner.scheduler._target_ else None
    if schlr:
        log.info("Scheduler loaded...")
    else:
        log.warning("Scheduler is not defined. Edit the configuration if this is not what you wanted.")
    
    # instantiate learner
    if 'path' in conf and conf.path:
        _mod, _bn = _dynamic_import(conf.path)
        learner = getattr(_mod, conf.cls)(model, loss, optim, schlr, resume_model, resume_opt, resume_dir)
    else:
        learner = getattr(import_module('.torch.learner', 'eniat'), conf.cls)(model, loss, optim, schlr, resume_model, resume_dir)
    if learner:
        log.info("Learner instance created.")

    return learner, None if not resume_opt else learner, ret_state

class TorchLearner(Learner, Generic[T_co]):
    def __init__(self, model:Module, criterion=None, optimizer=None, scheduler=None) -> None:
        super(TorchLearner).__init__()
        self.model = model
        self.loss_fn = criterion
        self.opt = optimizer
        self.sch = scheduler

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
                self.opt.load_state_dict(torch.load(f)['optimizer'])

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
    
    def to(self, device:Union[int, str]):
        self.model.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

class SupremeLearner (TorchLearner):
    def __init__(self, model: Module, criterion=None, optimizer=None, scheduler=None, resume: bool = False, resume_path: str = None) -> None:
        super().__init__(model, criterion, optimizer, scheduler)

    def fit(self, batch: Tensor, device: int, logger):
        x, y = batch
        x, y = x.to(device), y.to(device)
        model = self.model.to(device)
        return self.loss_fn(model(x), y)
    
    def predict(self, batch: Tensor, device: int, logger):
        batch = batch.to(device)
        model = self.model.to(device)
        return model(batch)