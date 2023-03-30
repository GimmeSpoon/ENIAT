from abc import abstractmethod, ABCMeta

class BaseLearner (metaclass=ABCMeta):
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