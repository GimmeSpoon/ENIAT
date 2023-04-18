from abc import ABCMeta, abstractmethod
from typing import TypeVar
from ..data.course import Course, FullCourse
from tqdm.auto import tqdm
from ..utils.grader import Grader
import sys
import warnings

D = TypeVar('D', bound=Course)

class Warning ():
    def __init__(self, logger) -> None:
        self.logger = logger

    def __call__(self, *args, **kwds):
        self.logger.warning(warnings.formatwarning(*args, **kwds))

class Learner (metaclass=ABCMeta):
    r'''Base class for all learners
    It is basically wrapper class for model such as module in pytorch or estimator in scikit-learn.
    This class is meant for just simple training and inference features, not for some other things like datastream manipulation.
    It is because the model could be used for inference only task in the future updates.'''

    @abstractmethod
    def predict(self, batch):
        raise NotImplementedError(f"'predict' method of {self.__class__.__name__} class is not implemented.")

    @abstractmethod
    def fit(self, batch):
        pass

    @abstractmethod
    def load_model(self): ...

L = TypeVar('L', bound=Learner)

class Trainer:
    r'''Base class for Trainer

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
    '''
    def __init__(self, course:D=None, learner:L=None, conf=None, grader=None, logger=None) -> None:
        self.course = course
        self.learner = learner
        self.conf = conf
        self.log = logger
        self.grader = grader

    def fit(self, silent:bool=False):
        self.course.select('train')
        for epoch in (epoch_bar:=tqdm( [1] if not self.conf.epoch else self.conf.epoch, desc='Training', unit='epoch', position=0, leave=False, disable=silent)):
            self.learner.fit(batch=self.course.next())

    def predict(self):
        self.course.select('predict')
        preds = self.learner.predict(batch=self.course.next())
        return preds

    def eval(self):
        self.course.select('eval')
        preds, gt = self.learner.eval(batch=self.course.next())
        if self.grader:
            return self.grader(preds, gt)
        else:
            return Grader.eval(preds, gt)