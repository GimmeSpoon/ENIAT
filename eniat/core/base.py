"""Base classes for basic components.
Each package-based components such as Trainer or Grader inherit classes from this module."""

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Callable, Sequence, Union, Any, Hashable
from .course import CourseBook
from ..utils import load_class
import copy
import warnings
from omegaconf import DictConfig

T_co = TypeVar("T_co", covariant=True)
B = TypeVar("B", bound=CourseBook)


class ConfigurationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Warning:
    def __init__(self, logger) -> None:
        self.logger = logger

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        self.logger.warning(
            warnings.formatwarning(message, category, filename, lineno, line)
        )


class Learner(metaclass=ABCMeta):
    r"""Base class for all learners
    It is basically wrapper class for model such as module in pytorch or estimator in scikit-learn.
    This class is meant for just simple training and inference features, not for some other things like datastream manipulation.
    It is because the model could be used for inference only task in the future updates."""

    @abstractmethod
    def predict(self, batch):
        raise NotImplementedError(
            f"'predict' method of {self.__class__.__name__} class is not implemented."
        )

    @abstractmethod
    def fit(self, batch):
        pass

    @abstractmethod
    def resume_model(self):
        ...


L = TypeVar("L", bound=Learner)


class Grader:
    class EvaluationResult:
        def __init__(self) -> None:
            self.__result = {}

        def done(self, data: Any, label: Hashable = None):
            if isinstance(data, dict):
                for key, value in data.items():
                    self.__result[key] = value
            else:
                self.__result[label] = data

        def as_dict(self) -> dict:
            return copy.deepcopy(self.__result)

        def __repr__(self) -> str:
            return "\n".join(
                [f"{label:20} : {res}" for label, res in self.__result.items()]
            )

    def __init__(
        self,
        conf: DictConfig,
        methods: Union[str, Callable, Sequence[Union[str, Callable]]] = None,
        options: list[dict] = None,
        logger=None,
        course: B = None,
        learner: L = None,
    ) -> None:
        self.methods = []
        self.conf = conf
        if conf.scheme.metrics:
            self.append_metric(conf.scheme.metrics)
        self.course = course
        self.learner = learner
        self.append_metric(methods)
        self.log = logger
        self.options = options

    def append_metric(self, methods: Union[Callable, Sequence[Callable]]):
        if callable(methods):
            self.methods.append(methods)
        elif methods is not None:
            for method in methods:
                if callable(method):
                    self.methods.append(method)
                else:
                    if isinstance(method, str):
                        self.methods.append(load_class(_class=method))
                    elif isinstance(method, DictConfig):
                        self.methods.append(
                            load_class(
                                method.path if "path" in method else None, method.cls
                            )
                        )
                    else:
                        raise ValueError("Invalid metric")

    def is_enabled(self) -> bool:
        return len(self.methods) > 0

    def compute(
        self, prediction: T_co, ground_truth: T_co, options: list[dict] = None
    ) -> dict:
        result = {}
        if options is None:
            options = self.options

        for i, method in enumerate(self.methods):
            result[method.__name__] = (
                method(prediction, ground_truth, options[i])
                if options and options[i]
                else method(prediction, ground_truth)
            )

        return result

    def __call__(self, prediction: T_co, ground_truth: T_co, options: list[dict]):
        self.compute(prediction, ground_truth)


class RemoteGrader:
    """Spawn a separated evalaution process"""

    def __init__(self) -> None:
        raise NotImplementedError()

    def run() -> None:
        pass


class Trainer:
    r"""Base class for Trainer

    Trainer provides logging and saving checkpoints, furthermore lessen the repetitive parts in usual machine learning codes.
    BaseTrainer provides fit, eval, and predict features even it not specifying compatibility with libraries such as PyTorch or Scikit-learn.
    It can work with some of simple models, thus you can use this standalone. But for supported libraries, it would be recommended to use based class for the task.
    Remember, Trainers are designed to do only one task at a time, which means for multiple tasks you are required to initiate the same number of trainers.
    This is beacause for distributed learning in PyTorch, it would complicate the whole project if the trainer manages multiple tasks by alone.

    Arguments:
        :param course: dataset for the task. It should not be modified while on doing tasks.
        :param learner: learner(model) for the task. recommend to use same based class e.g. TorchLearner for TorchTrainer.
        :param conf: Trainer configuration structured by omegaconf DictConfig.
        :param grader: Evalutor from eniat package.
        :param logger: Logger from eniat package. required.
    """

    def __init__(
        self, course: B = None, learner: L = None, conf=None, grader=None, logger=None
    ) -> None:
        self.course = course
        self.learner = learner
        self.conf = conf
        self.grader = grader
        self.log = logger

    @abstractmethod
    def fit(self, silent: bool = False, eval: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()
