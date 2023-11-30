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

class TorchLearner(Learner, Generic[T_co]):
    def __init__(
        self,
        conf: DictConfig = None,
        model: Module = None,
    ) -> None:
        super(TorchLearner).__init__()
        self.conf = conf
        self.model = model

    @abstractmethod
    def fit(self, batch: Tensor, device: int, logger):
        pass

    @abstractmethod
    def predict(self, batch: Tensor, device: int, logger):
        pass

    def prepare(self, log: L = None) -> None:
        self.model = instantiate(
            self.conf.model
        )

    def resume_model(self, state=None, path: Union[str, Path] = None) -> None:
        if state:
            self.model.load_state_dict(state)
        else:
            with open(path, "rb") as f:
                self.model.load_state_dict(torch.load(f))

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

    def to(self, device: Union[int, str]):
        self.model.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)


class SupervisedLearner(TorchLearner):
    def __init__(
        self, conf:DictConfig, model: Module = None,
    ) -> None:
        super().__init__(conf, model)

    def fit(self, batch: Tensor, device: int, logger):
        x = batch
        x = x.to(device)
        model = self.model.to(device)
        return model(x)

    def predict(self, batch: Tensor, device: int, logger):
        batch = batch.to(device)
        self.model.to(device)
        return self.model(batch)
