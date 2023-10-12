from typing import TypeVar
from abc import abstractmethod
from ..core import Learner
from sklearn.base import BaseEstimator


B = TypeVar("B", bound=BaseEstimator)


class ScikitLearner(Learner):
    def __init__(self, model: B) -> None:
        super(ScikitLearner).__init__()

        self.model = model

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit_step(self):
        pass

    def get_model(self) -> B:
        return self.model
