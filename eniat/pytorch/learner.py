from typing import TypeVar, Generic
from abc import abstractmethod
from ..base import Learner
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

T_co = TypeVar('T_co', covariant=True)
O = TypeVar('O', bound=Optimizer)

class TorchLearner(Learner, Generic[T_co]):
    def __init__(self, model:Module, criterion=None, optimizer=None, scheduler=None, resume:bool=False, resume_path:str=None) -> None:
        super(TorchLearner).__init__()
        self.model = model
        self.loss_fn = criterion
        self.opt = optimizer
        self.sch = scheduler

        if resume:
            self.load_model(resume_path)

    @abstractmethod
    def fit(self, batch:Tensor, device:int, logger):
        pass
    
    @abstractmethod
    def predict(self, batch:Tensor, device:int, logger):
        pass

    def save_model(self, path:str) -> None:   
        torch.save(self.model.state_dict(), path)

    def save_optimizer(self, path:str) -> None:
        torch.save(self.opt.state_dict(), path)
    
    def save_all (self, model_path:str, optimizer_path:str) -> None:
        self.save_model(model_path)
        self.save_optimizer(optimizer_path)

    def load_model(self, path:str, state=None) -> None:
        if state:
            self.model.load_state_dict(state)
        else:
            with open(path, 'rb') as f:
                self.model.load_state_dict(torch.load(f))

    def load_optimizer(self, path:str, state=None) -> None:
        if state:
            self.opt.load_state_dict(state)
        else:
            with open(path, 'rb') as f:
                self.opt.load_state_dict(torch.load(f))

class SupremeLearner (TorchLearner):
    def __init__(self, model: Module, criterion=None, optimizer=None, scheduler=None, resume: bool = False, resume_path: str = None) -> None:
        super().__init__(model, criterion, optimizer, scheduler, resume, resume_path)

    def fit(self, batch: Tensor, device: int, logger):
        return super().fit(batch, device, logger)
    
    def predict(self, batch: Tensor, device: int, logger):
        return super().predict(batch, device, logger)