from typing import TypeVar
from eniat.learner import BaseLearner
from eniat.data.course import Course, FullCourse
from tqdm.auto import tqdm
from eniat.utils.grader import Grader

L = TypeVar('L', bound=BaseLearner)
D = TypeVar('D', bound=Course)

class BaseTrainer:
    r'''Base class for Trainer
    Trainer gets a data and model pair to train or validate.
    For the purpose, trainer will manage environments for learners.
    Remember, Trainers can have multiple models learned or infer but must have only unique dataset pair.
    '''
    def __init__(self, course:D=None, learner:L=None, trainer_conf=None, evaluate_fn=None, logger=None) -> None:
        self.learner = learner
        if isinstance(course, Course):
            self.course = FullCourse(course)
        else:
            self.course = course
        self.conf = trainer_conf
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