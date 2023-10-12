from typing import Literal, TypeVar
from . import Logger
from omegaconf import DictConfig

L = TypeVar("L", bound=Logger)


class Manager:
    r"""Manager scary ever"""

    def __init__(self) -> None:
        pass

    def run(self):
        raise NotImplementedError("sorry dude")

    @staticmethod
    def load_componenets(package: Literal["torch", "sklearn"], conf: DictConfig):
        if package == "torch":
            return Manager.load_torch()
        elif package == "sklearn":
            return Manager.load_sklearn()
        else:
            raise ValueError("")

    @staticmethod
    def load_torch(task: Literal["fit", "eval", "infer"], conf: DictConfig, log: L):
        from ..torch.learner import instantiate_learner
        from ..torch.grader import load_grader
        from ..torch.trainer import load_trainer

        learner = instantiate_learner(conf.learner, log)
        if task == "fit":
            learner = instantiate_learner(
                conf.learner,
                log,
            )

    @staticmethod
    def load_sklearn():
        raise NotImplementedError("sorry dude")
