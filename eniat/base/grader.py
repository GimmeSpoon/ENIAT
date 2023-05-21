from typing import Sequence, Union, TypeVar, Callable, Literal
from numpy import ndarray
import sklearn.metrics as mt
from . import Learner
from ..data import Course
from omegaconf import DictConfig

T_co = TypeVar('T_co', covariant=True)
L = TypeVar('L', bound=Learner)
C = TypeVar('C', bound=Course)

sk_metric = {
    "acc" : mt.accuracy_score,
    "f1" : mt.f1_score,
    "top_k_acc" : mt.top_k_accuracy_score,
    "fbeta" : mt.fbeta_score,
    "hamming" : mt.hamming_loss,
    "hinge" : mt.hinge_loss,
    "jaccard" : mt.jaccard_score,
    "log" : mt.log_loss,
    "precision" : mt.precision_score,
    "recall" : mt.recall_score,
    "roc_auc" : mt.roc_auc_score,
    "pr_curve" : mt.precision_recall_curve,
    "roc_curve" : mt.roc_curve,
    "01" : mt.zero_one_loss,
    "max" : mt.max_error,
    "mae" : mt.mean_absolute_error,
    "mse" : mt.mean_squared_error,
    "msle" : mt.mean_squared_log_error,
    "meae" : mt.median_absolute_error,
}

def sk_eval(preds, gt, methods:Union[str, Sequence[str]], **kwargs):
        
    if isinstance(method, str):
        methods = [method]

    if not isinstance(preds, ndarray):
        preds = preds.numpy()

    if not isinstance(gt, ndarray):
        gt = gt.numpy()

    ret = []

    for method in methods:
        ret.append(sk_metric[method](gt, preds, **kwargs))

    return ret

class Grader():
    
    class Result():
        def __init__(self) -> None:
            pass
    
    def __init__(self, conf:DictConfig, methods:Union[str, Callable, Sequence[Union[str, Callable]]], logger, course:C=None, options:list[dict]=None) -> None:
        self.conf = conf
        if conf.methods:
            self.append_metric(conf.methods)
        self.unit = conf.unit
        self.interval = conf.interval
        self.course = course
        self.methods = []
        self.append_metric(methods)
        self.log = logger
        self.options = options

    def append_metric(self, method:Union[str, Callable, Sequence[Union[str, Callable]]]):
        if isinstance(methods, Callable):
            methods = [methods]
        elif isinstance(methods, str):
            methods = [sk_metric[methods]]
        else:
            methods = []
            for method in methods:
                if isinstance(methods, Callable):
                    methods.append(method)
                else:
                    methods.append(sk_metric[methods])

    def compute(self, prediction:T_co, ground_truth:T_co, options:list[dict]=None) -> dict:
        result = {}
        if options is None:
            options = self.options
        opt = 0
        for method in self.methods:
            result[method.__name__] = method(prediction, ground_truth, options[opt]) if options and options[opt] else \
            method(prediction, ground_truth)
            opt += 1

        self.log.log_state()

        return result
    
    def __call__(self, prediction:T_co, ground_truth:T_co, options:list[dict]):
        self.compute(prediction, ground_truth)

class RemoteGrader():
    '''Spawn a separated evalaution process'''
    def __init__(self) -> None:
        pass

    def run() -> None:
        pass