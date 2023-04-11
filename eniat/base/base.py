from abc import ABCMeta, abstractmethod
from typing import TypeVar
from ..data.course import Course, FullCourse
from tqdm.auto import tqdm
from ..utils.grader import Grader

D = TypeVar('D', bound=Course)

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
    def save_model(self): ...

    @abstractmethod
    def load_model(self): ...

L = TypeVar('L', bound=Learner)

class Trainer:
    r'''Base class for Trainer
    Trainer gets a data and model pair to train or validate.
    For the purpose, trainer will manage environments for learners.
    Remember, Trainers can have multiple models learned or infer but must have only unique dataset pair.
    '''
    def __init__(self, course:D=None, learner:L=None, conf=None, evaluate_fn=None, logger=None) -> None:
        self.learner = learner
        if isinstance(course, Course):
            self.course = FullCourse(course)
        else:
            self.course = course
        self.conf = conf
        self.logger = logger

        self.eval_fn = evaluate_fn

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
        if self.eval_fn:
            return self.eval_fn(preds, gt)
        else:
            return Grader.eval(preds, gt)