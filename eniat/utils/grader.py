from typing import Sequence, Union, TypeVar, Callable
from numpy import ndarray
import sklearn.metrics as mt
from omegaconf import DictConfig

T_co = TypeVar('T_co', covariant=True)

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
    "mdae" : mt.median_absolute_error,
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
    def __init__(self, conf:DictConfig, methods:Union[Callable, Sequence[Callable]]) -> None:
        self.unit = conf.unit
        self.interval = conf.interval
        if isinstance(methods, Callable):
            methods = [methods]
        self.methods = list(methods)

    def append_metric(self, method:Callable):
        self.methods.append(method)

    def _stepfilter(fn:Callable) -> Callable:
        def wrapper(self, pred, gt, timestep:int, unit:str, force:bool=False):
            if not force and (self.unit != unit or timestep % self.interval):
                return
            return fn(self, pred, gt)
        return wrapper

    @_stepfilter
    def compute(self, prediction:T_co, ground_truth:T_co) -> dict:
        result = {}
        for method in self.methods:
            result[method.__name__]=method(prediction, ground_truth)
        return result

    @_stepfilter
    def __call__(self, prediction:T_co, ground_truth:T_co):
        self.compute(prediction, ground_truth)