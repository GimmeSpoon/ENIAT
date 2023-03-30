from typing import TypeVar
from abc import abstractmethod
from BaseLearner import BaseLearner
from sklearn.base import BaseEstimator


B = TypeVar('B', bound=BaseEstimator)

class ScikitLearner (BaseLearner):
    def __init__(self, model:B) -> None:
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